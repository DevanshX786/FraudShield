"""Integration tests for FastAPI Serving and Fallback paths."""

import datetime
import json
import os
import shutil
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Table,
    create_engine,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

# Set DATABASE_URL env var before importing app
testing_db_file = Path("test_api.db")
os.environ["DATABASE_URL"] = f"sqlite:///{testing_db_file}"

import src.api.database as api_db  # noqa: E402
from src.api.database import Base  # noqa: E402
from src.api.main import app, get_db  # noqa: E402
import src.api.main as api_main  # noqa: E402
from src.api.models import (  # noqa: E402
    FeatureRecord,
    PredictionRecord,
    DriftLogRecord,
    RetrainLogRecord,
)

# Initialize test engine and SessionLocal with names safe from pytest collection
db_engine_local = create_engine(
    f"sqlite:///{testing_db_file}",
    connect_args={"check_same_thread": False},
)
session_factory_local = sessionmaker(
    autocommit=False, autoflush=False, bind=db_engine_local
)

# Overwrite database.py references
api_db.engine = db_engine_local
api_db.SessionLocal = session_factory_local

# Register model_registry table on Base.metadata
_ = Table(
    "model_registry",
    Base.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("model_name", String),
    Column("model_version", String),
    Column("run_id", String),
    Column("f1_score", Float),
    Column("auc_score", Float),
    Column("precision_score", Float),
    Column("recall_score", Float),
    Column("threshold", Float),
    Column("is_production", Boolean),
    Column("created_at", DateTime, default=func.now()),
    extend_existing=True,
)


class MockModel:
    def __init__(self, prob=0.1):
        self.prob = prob

    def predict_proba(self, X):
        return np.array([[1.0 - self.prob, self.prob]])


class MockExplainer:
    def shap_values(self, X):
        return np.array([[0.05] * X.shape[1]])


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_db_file():
    """Ensure test_api.db is cleaned up at the end of the test session."""
    yield
    if testing_db_file.exists():
        try:
            testing_db_file.unlink()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def clean_logs_and_models_dirs():
    """Backup any existing logs and models file, clean up workspace, restore later."""
    backed_up = {}
    paths_to_clean = [
        Path("logs/features.json"),
        Path("logs/predictions.json"),
        Path("logs/drift_logs.json"),
        Path("logs/retraining_history.json"),
        Path("models/model_registry_metadata.json"),
    ]
    for path in paths_to_clean:
        if path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
            backed_up[path] = backup_path
            path.unlink()
    yield
    for path in paths_to_clean:
        if path.exists():
            path.unlink()
    for orig_path, backup_path in backed_up.items():
        shutil.move(backup_path, orig_path)


@pytest.fixture(autouse=True)
def mock_ml_components(monkeypatch):
    """Setup default mocks for Model, Explainer, preprocessing, and retraining."""
    model = MockModel()
    explainer = MockExplainer()
    monkeypatch.setattr(api_main, "MODEL", model)
    monkeypatch.setattr(api_main, "EXPLAINER", explainer)
    monkeypatch.setattr(api_main, "MODEL_SOURCE", "mock_source")
    monkeypatch.setattr(api_main, "MODEL_VERSION", "mock_v1")

    def mock_preprocess(df, is_train=False):
        return pd.DataFrame({"feat_1": [0.5], "feat_2": [-1.0]})

    monkeypatch.setattr("src.api.main.preprocess_data", mock_preprocess)

    def mock_retrain():
        pass

    monkeypatch.setattr("src.api.main.retrain_pipeline", mock_retrain)


@pytest.fixture(name="db_session")
def db_session_fixture():
    """SQLite file session fixture with all schemas."""
    Base.metadata.create_all(bind=db_engine_local)
    session = session_factory_local()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=db_engine_local)


@pytest.fixture(name="client")
def client_fixture(db_session):
    """API client using overridden database connection."""

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(name="offline_client")
def offline_client_fixture(monkeypatch):
    """API client configured to simulate offline database."""

    def override_get_db():
        yield None

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr("src.api.main.check_db_health", lambda: False)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_online_model_loaded(client, monkeypatch) -> None:
    """Test health check when database is online and model is loaded."""
    monkeypatch.setattr("src.api.main.check_db_health", lambda: True)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "connected",
        "model": "loaded",
        "model_source": "mock_source",
    }


def test_health_offline_model_missing(offline_client, monkeypatch) -> None:
    """Test health check when database is offline and model is missing."""
    monkeypatch.setattr(api_main, "MODEL", None)
    monkeypatch.setattr(api_main, "MODEL_SOURCE", "none")
    response = offline_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "database": "offline",
        "model": "missing",
        "model_source": "none",
    }


def test_predict_endpoint_clean(client, db_session) -> None:
    """Test predict endpoint for clean transaction (prob < 0.3)."""
    payload = {
        "transaction_id": "tx_001",
        "transaction_amt": 50.0,
        "product_cd": "W",
        "card1": 1000.0,
        "card2": 200.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 226.0,
        "card6": "debit",
        "addr1": 321.0,
        "P_emaildomain": "gmail.com",
        "transaction_dt": 86400,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx_001"
    assert data["prediction"] == "CLEAN"
    assert "fraud_probability" in data
    assert "shap_explanation" in data
    assert "feat_1" in data["shap_explanation"]

    # Verify features and predictions are saved in DB
    feat_records = db_session.query(FeatureRecord).all()
    assert len(feat_records) == 1
    assert feat_records[0].transaction_id == "tx_001"

    pred_records = db_session.query(PredictionRecord).all()
    assert len(pred_records) == 1
    assert pred_records[0].transaction_id == "tx_001"


def test_predict_endpoint_suspicious(client, monkeypatch) -> None:
    """Test predict endpoint for suspicious transaction (0.3 <= prob < 0.5)."""
    monkeypatch.setattr(api_main.MODEL, "prob", 0.4)
    payload = {
        "transaction_id": "tx_002",
        "transaction_amt": 150.0,
        "product_cd": "W",
        "transaction_dt": 86500,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "SUSPICIOUS"
    assert data["confidence"] == "MEDIUM"


def test_predict_endpoint_fraud(client, monkeypatch) -> None:
    """Test predict endpoint for fraud transaction (prob >= 0.5)."""
    monkeypatch.setattr(api_main.MODEL, "prob", 0.85)
    payload = {
        "transaction_id": "tx_003",
        "transaction_amt": 500.0,
        "product_cd": "C",
        "transaction_dt": 86600,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "FRAUD"
    assert data["confidence"] == "HIGH"


def test_predict_offline_fallback(offline_client) -> None:
    """Test predict fallback to JSON storage when database is offline."""
    payload = {
        "transaction_id": "tx_offline_001",
        "transaction_amt": 99.0,
        "product_cd": "W",
        "transaction_dt": 90000,
    }
    response = offline_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx_offline_001"

    # Verify fallback files were created
    features_json = Path("logs/features.json")
    predictions_json = Path("logs/predictions.json")
    assert features_json.exists()
    assert predictions_json.exists()

    with open(predictions_json, "r") as f:
        preds = json.load(f)
        assert len(preds) == 1
        assert preds[0]["transaction_id"] == "tx_offline_001"


def test_metrics_online(client, db_session) -> None:
    """Test retrieval of model performance metrics from database."""
    db_session.execute(
        Base.metadata.tables["model_registry"]
        .insert()
        .values(
            model_name="test_model",
            model_version="mock_v1",
            run_id="run_123",
            f1_score=0.82,
            auc_score=0.88,
            precision_score=0.84,
            recall_score=0.80,
            threshold=0.5,
            is_production=True,
        )
    )
    db_session.commit()
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json() == {
        "f1_score": 0.82,
        "auc_score": 0.88,
        "precision_score": 0.84,
        "recall_score": 0.80,
    }


def test_metrics_offline(offline_client) -> None:
    """Test retrieval of model performance metrics from local registry fallback."""
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    local_registry_file = models_dir / "model_registry_metadata.json"
    with open(local_registry_file, "w") as f:
        json.dump(
            [
                {
                    "is_production": True,
                    "f1_score": 0.79,
                    "auc_score": 0.85,
                    "precision_score": 0.81,
                    "recall_score": 0.77,
                }
            ],
            f,
        )
    response = offline_client.get("/metrics")
    assert response.status_code == 200
    assert response.json() == {
        "f1_score": 0.79,
        "auc_score": 0.85,
        "precision_score": 0.81,
        "recall_score": 0.77,
    }


def test_drift_online(client, db_session) -> None:
    """Test retrieval of data drift status from database."""
    run_date = datetime.date(2026, 6, 18)
    db_session.add(
        DriftLogRecord(
            run_date=run_date,
            feature_name="amount",
            drift_score=0.04,
            p_value=0.01,
            drift_detected=True,
            drift_intensity=0.5,
        )
    )
    db_session.commit()

    response = client.get("/drift")
    assert response.status_code == 200
    assert response.json()["drift_detected"] is True


def test_drift_history_online(client, db_session) -> None:
    """Test retrieval of data drift history from database."""
    run_date = datetime.date(2026, 6, 18)
    db_session.add(
        DriftLogRecord(
            run_date=run_date,
            feature_name="amount",
            drift_score=0.04,
            p_value=0.01,
            drift_detected=True,
            drift_intensity=0.5,
        )
    )
    db_session.commit()

    response = client.get("/drift/history")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_drift_offline(offline_client) -> None:
    """Test retrieval of data drift status from local file fallback."""
    drift_logs_file = Path("logs/drift_logs.json")
    with open(drift_logs_file, "w") as f:
        json.dump(
            [
                {
                    "run_date": "2026-06-18",
                    "feature_name": "velocity_1hr",
                    "drift_score": 0.08,
                    "p_value": 0.002,
                    "drift_detected": True,
                    "drift_intensity": 0.6,
                }
            ],
            f,
        )
    response = offline_client.get("/drift")
    assert response.json()["drift_detected"] is True


def test_drift_history_offline(offline_client) -> None:
    """Test retrieval of data drift history from local file fallback."""
    drift_logs_file = Path("logs/drift_logs.json")
    with open(drift_logs_file, "w") as f:
        json.dump(
            [
                {
                    "run_date": "2026-06-18",
                    "feature_name": "velocity_1hr",
                    "drift_score": 0.08,
                    "p_value": 0.002,
                    "drift_detected": True,
                    "drift_intensity": 0.6,
                }
            ],
            f,
        )
    response = offline_client.get("/drift/history")
    assert len(response.json()) == 1
    assert response.json()[0]["feature_name"] == "velocity_1hr"


def test_model_info_online(client, db_session) -> None:
    """Test model version and promotion history metadata from database."""
    db_session.add(
        RetrainLogRecord(
            triggered_at=datetime.datetime(2026, 6, 18, 12, 0, 0),
            trigger_reason="drift",
            old_f1=0.80,
            new_f1=0.83,
            promoted=True,
            notes="Promoted successfully",
        )
    )
    db_session.commit()

    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "mock_v1"
    assert len(data["promotion_history"]) == 1


def test_model_info_offline(offline_client) -> None:
    """Test model version and promotion history metadata from file fallback."""
    retrain_history_file = Path("logs/retraining_history.json")
    with open(retrain_history_file, "w") as f:
        json.dump(
            [
                {
                    "triggered_at": "2026-06-18T12:00:00",
                    "trigger_reason": "scheduled",
                    "old_f1": 0.78,
                    "new_f1": 0.79,
                    "promoted": False,
                    "notes": "Insufficient improvement",
                }
            ],
            f,
        )
    response = offline_client.get("/model/info")
    data = response.json()
    assert len(data["promotion_history"]) == 1
    assert data["promotion_history"][0]["trigger_reason"] == "scheduled"


def test_recent_transactions_online(client, db_session) -> None:
    """Test fetching recent predictions log from database."""
    db_session.add(
        PredictionRecord(
            transaction_id="tx_recent_db",
            fraud_probability=0.02,
            prediction="CLEAN",
            model_version="mock_v1",
            shap_values={"feat_1": 0.0},
        )
    )
    db_session.commit()

    response = client.get("/transactions/recent")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["transaction_id"] == "tx_recent_db"


def test_recent_transactions_offline(offline_client) -> None:
    """Test fetching recent predictions log from file fallback."""
    predictions_file = Path("logs/predictions.json")
    with open(predictions_file, "w") as f:
        json.dump(
            [
                {
                    "transaction_id": "tx_recent_json",
                    "fraud_probability": 0.95,
                    "prediction": "FRAUD",
                    "model_version": "mock_v1",
                    "shap_explanation": {"feat_1": 0.3},
                }
            ],
            f,
        )
    response = offline_client.get("/transactions/recent")
    assert len(response.json()) == 1
    assert response.json()[0]["transaction_id"] == "tx_recent_json"


def test_grafana_metrics(client, db_session) -> None:
    """Test formatting time-series metrics for Grafana Cloud."""
    db_session.add(
        PredictionRecord(
            transaction_id="tx_grafana",
            fraud_probability=0.9,
            prediction="FRAUD",
            model_version="mock_v1",
            shap_values={},
        )
    )
    db_session.commit()

    response = client.get("/grafana-metrics")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 4
    targets = [metric["target"] for metric in data]
    assert "fraud_rate" in targets
    assert "total_predictions" in targets
    assert "drift_detected" in targets
    assert "avg_latency_ms" in targets


def test_trigger_manual_retrain(client) -> None:
    """Test asynchronous background retraining trigger."""
    response = client.post("/retrain")
    assert response.status_code == 200
    assert response.json() == {"status": "retraining started"}


def test_predict_endpoint_missing_model(client, monkeypatch) -> None:
    """Test predict endpoint throws 503 if model is missing/None."""
    monkeypatch.setattr(api_main, "MODEL", None)
    payload = {
        "transaction_id": "tx_no_model",
        "transaction_amt": 50.0,
        "product_cd": "W",
        "transaction_dt": 100,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert "model is not loaded" in response.json()["detail"]


def test_recent_transactions_offline_with_features(offline_client) -> None:
    """Test fetching recent predictions log from file fallback with amount/card_type join."""
    predictions_file = Path("logs/predictions.json")
    features_file = Path("logs/features.json")

    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "transaction_id": "tx_with_feat",
                    "fraud_probability": 0.05,
                    "prediction": "CLEAN",
                    "model_version": "mock_v1",
                    "shap_explanation": {"feat_1": 0.01},
                }
            ],
            f,
        )
    with open(features_file, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "transaction_id": "tx_with_feat",
                    "amount": 250.50,
                    "card_type": "credit",
                    "card_network": "mastercard",
                }
            ],
            f,
        )

    response = offline_client.get("/transactions/recent")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["transaction_id"] == "tx_with_feat"
    assert data[0]["amount"] == 250.50
    assert data[0]["card_type"] == "credit (mastercard)"
