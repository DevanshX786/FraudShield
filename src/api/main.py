"""FastAPI backend application for FraudShield serving and monitoring."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
from contextlib import asynccontextmanager
import pandas as pd
import shap
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.database import Base, check_db_health, engine, get_db
from src.api.models import (
    DriftLogRecord,
    FeatureRecord,
    PredictionRecord,
    RetrainLogRecord,
)
from src.data_preprocessing import preprocess_data
from src.retraining_pipeline import retrain_pipeline

# Load configuration
CONFIG_PATH = "config/config.yaml"
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        import yaml

        config = yaml.safe_load(f)
except Exception:
    config = {}

# Global model and explainer references loaded at startup
MODEL = None
MODEL_SOURCE = "none"
MODEL_VERSION = "unknown"
EXPLAINER = None


def load_production_model() -> None:
    """Load production model from MLflow registry or local fallback."""
    global MODEL, MODEL_SOURCE, MODEL_VERSION, EXPLAINER
    # Configure MLflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)

    model_name = os.getenv("MODEL_NAME", "fraudshield-xgboost")
    # 1. Try MLflow Registry
    try:
        model_uri = f"models://{model_name}@Production"
        MODEL = mlflow.xgboost.load_model(model_uri)
        MODEL_SOURCE = "mlflow_production"
        # Get model version
        client = mlflow.tracking.MlflowClient()
        try:
            model_version_details = client.get_model_version_by_alias(
                model_name, "Production"
            )
            MODEL_VERSION = f"v{model_version_details.version}"
        except Exception:
            versions = client.search_model_versions(f"name='{model_name}'")
            for v in versions:
                if getattr(
                    v, "current_stage", None
                ) == "Production" or "Production" in getattr(v, "aliases", []):
                    MODEL_VERSION = f"v{v.version}"
                    break
        print(f"Loaded production model from MLflow: {MODEL_VERSION}")
    except Exception as e:
        print(f"MLflow model load failed: {e}. Falling back to local model file...")

    # 2. Try Local fallback
    if MODEL is None:
        try:
            local_model_path = Path("models/xgboost_model.joblib")
            if local_model_path.exists():
                MODEL = joblib.load(local_model_path)
                MODEL_SOURCE = "local_fallback"
                MODEL_VERSION = "local_v1"
                print(
                    f"Loaded production model from local file: " f"{local_model_path}"
                )
            else:
                print(
                    "Warning: No local model file found at "
                    "models/xgboost_model.joblib"
                )
        except Exception as ex:
            print(f"Error loading local model file: {ex}")

    # Initialize SHAP explainer if model is loaded
    if MODEL is not None:
        try:
            EXPLAINER = shap.TreeExplainer(MODEL)
        except Exception as shap_ex:
            print(f"Warning: Failed to initialize SHAP " f"TreeExplainer: {shap_ex}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables and load the production model."""
    if engine is not None:
        try:
            Base.metadata.create_all(bind=engine)
            print("Successfully initialized PostgreSQL database schemas.")
        except Exception as e:
            print(f"Warning: Database schema initialization failed: {e}")

    load_production_model()
    yield


app = FastAPI(title="FraudShield API", version="1.0.0", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionPayload(BaseModel):
    """Pydantic schema for realtime prediction requests."""

    transaction_id: str
    transaction_amt: float
    product_cd: str
    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    transaction_dt: int


# Local memory cache for features history when DB is offline
LOCAL_FEATURES_CACHE: list[dict[str, Any]] = []
LOCAL_PREDICTIONS_CACHE: list[dict[str, Any]] = []

# In-memory rolling latency history (last 1000 requests)
LATENCY_HISTORY: list[float] = []


@app.middleware("http")
async def log_latency(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    # Exclude internal metrics and health paths to avoid polluting average latency
    path = request.url.path
    if not path.endswith("/grafana-metrics") and not path.endswith("/health"):
        global LATENCY_HISTORY
        LATENCY_HISTORY.append(duration_ms)
        if len(LATENCY_HISTORY) > 1000:
            LATENCY_HISTORY.pop(0)

    return response


def get_historical_features(
    user_key: str, merchant_key: str, db: Session | None
) -> list[dict[str, Any]]:
    """Retrieve historical transaction features from DB or local JSON cache."""
    history = []

    # 1. Try DB
    if db is not None:
        try:
            records = (
                db.query(FeatureRecord)
                .filter(
                    (FeatureRecord.user_key == user_key)
                    | (FeatureRecord.merchant_key == merchant_key)
                )
                .all()
            )
            for r in records:
                history.append(
                    {
                        "user_key": r.user_key,
                        "merchant_key": r.merchant_key,
                        "transaction_dt": (
                            r.hour * 3600 + (r.is_weekend * 86400)
                        ),  # Approximate
                        "amount": r.amount,
                    }
                )
            return history
        except Exception as e:
            print(f"Warning: Failed to query historical features from DB: {e}")

    # 2. Fallback to Local cache
    global LOCAL_FEATURES_CACHE
    history_file = Path("logs/features.json")
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                LOCAL_FEATURES_CACHE = json.load(f)
        except Exception:
            pass

    for r in LOCAL_FEATURES_CACHE:
        if r.get("user_key") == user_key or r.get("merchant_key") == merchant_key:
            history.append(r)

    return history


def save_feature_record(record: dict[str, Any], db: Session | None) -> None:
    """Save engineered feature record to database or local fallback JSON."""
    # 1. Try DB
    if db is not None:
        try:
            db_record = FeatureRecord(
                transaction_id=record["transaction_id"],
                amount=record["amount"],
                hour=record["hour"],
                day_of_week=record["day_of_week"],
                is_weekend=record["is_weekend"],
                user_key=record["user_key"],
                merchant_key=record["merchant_key"],
                user_tx_count_30d=record["user_tx_count_30d"],
                user_avg_amount_30d=record["user_avg_amount_30d"],
                merchant_tx_count_30d=record["merchant_tx_count_30d"],
                velocity_1hr=record["velocity_1hr"],
                card_network=record["card_network"],
                card_type=record["card_type"],
            )
            db.merge(db_record)
            db.commit()
            return
        except Exception as e:
            print(f"Warning: Failed to log feature record to database: {e}")

    # 2. Fallback to Local JSON
    global LOCAL_FEATURES_CACHE
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    history_file = logs_dir / "features.json"

    # Add to in-memory list and write
    LOCAL_FEATURES_CACHE.append(record)
    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(LOCAL_FEATURES_CACHE, f, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save local features cache file: {e}")


def save_prediction_record(record: dict[str, Any], db: Session | None) -> None:
    """Save realtime prediction metadata to database or local fallback JSON."""
    # 1. Try DB
    if db is not None:
        try:
            db_record = PredictionRecord(
                transaction_id=record["transaction_id"],
                fraud_probability=record["fraud_probability"],
                prediction=record["prediction"],
                model_version=record["model_version"],
                shap_values=record["shap_explanation"],
            )
            db.add(db_record)
            db.commit()
            return
        except Exception as e:
            print(f"Warning: Failed to log prediction to database: {e}")

    # 2. Fallback to Local JSON
    global LOCAL_PREDICTIONS_CACHE
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = logs_dir / "predictions.json"

    if predictions_file.exists():
        try:
            with open(predictions_file, "r", encoding="utf-8") as f:
                LOCAL_PREDICTIONS_CACHE = json.load(f)
        except Exception:
            pass

    LOCAL_PREDICTIONS_CACHE.append(record)
    try:
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(LOCAL_PREDICTIONS_CACHE, f, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save local predictions file: {e}")


@app.post("/predict")
def predict(
    payload: TransactionPayload, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Calculate fraud probability and SHAP values for an incoming transaction."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Production model is not loaded. Please train a model first.",
        )

    # 1. Build Entity proxies
    user_key = (
        f"{payload.card1}_{payload.card2}_{payload.card3}_"
        f"{payload.card5}_{payload.addr1}"
    )
    merchant_key = f"{payload.product_cd}_{payload.card4}_{payload.P_emaildomain}"

    # 2. Load history & compute point-in-time rolling aggregates
    new_dt = payload.transaction_dt
    new_amount = payload.transaction_amt
    history = get_historical_features(user_key, merchant_key, db)

    user_hist = [
        h
        for h in history
        if h.get("user_key") == user_key and h.get("transaction_dt", 0) < new_dt
    ]
    user_30d = [
        h for h in user_hist if h.get("transaction_dt", 0) >= new_dt - 30 * 86400
    ]
    user_1h = [h for h in user_hist if h.get("transaction_dt", 0) >= new_dt - 3600]

    user_tx_count_30d = len(user_30d) + 1
    user_total_amount_30d = sum(h.get("amount", 0.0) for h in user_30d) + new_amount
    user_avg_amount_30d = user_total_amount_30d / user_tx_count_30d
    velocity_1hr = len(user_1h) + 1

    merch_hist = [
        h
        for h in history
        if h.get("merchant_key") == merchant_key and h.get("transaction_dt", 0) < new_dt
    ]
    merch_30d = [
        h for h in merch_hist if h.get("transaction_dt", 0) >= new_dt - 30 * 86400
    ]

    merchant_tx_count_30d = len(merch_30d) + 1
    merchant_unique_users_30d = len(
        set(h.get("user_key") for h in merch_30d) | {user_key}
    )

    # Temporal feature computations
    day_number = new_dt // 86400
    hour = int((new_dt % 86400) // 3600)
    day_of_week = int(day_number % 7)
    is_weekend = bool(day_of_week in [5, 6])
    month = int(day_number // 30)

    # Compile engineered dict
    engineered_dict = {
        "transaction_id": payload.transaction_id,
        "TransactionID": payload.transaction_id,
        "amount": new_amount,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "user_key": user_key,
        "merchant_key": merchant_key,
        "user_tx_count_30d": user_tx_count_30d,
        "user_total_amount_30d": user_total_amount_30d,
        "user_avg_amount_30d": user_avg_amount_30d,
        "merchant_tx_count_30d": merchant_tx_count_30d,
        "merchant_unique_users_30d": merchant_unique_users_30d,
        "velocity_1hr": velocity_1hr,
        "card_network": payload.card4 or "unknown",
        "card_type": payload.card6 or "unknown",
        "card1": payload.card1,
        "card2": payload.card2,
        "card3": payload.card3,
        "card5": payload.card5,
        "addr1": payload.addr1,
        "ProductCD": payload.product_cd,
        "card4": payload.card4 or "unknown",
        "card6": payload.card6 or "unknown",
        "P_emaildomain": payload.P_emaildomain or "unknown",
        "R_emaildomain": payload.R_emaildomain or "unknown",
        "DeviceType": payload.DeviceType or "unknown",
        "TransactionDT": new_dt,
        "day_number": day_number,
        "month": month,
    }

    # Save to features history
    save_feature_record(engineered_dict, db)

    # 3. Preprocess single transaction row
    feature_df = pd.DataFrame([engineered_dict])
    keep_cols = [
        "TransactionID",
        "TransactionDT",
        "amount",
        "day_number",
        "hour",
        "day_of_week",
        "is_weekend",
        "month",
        "user_key",
        "merchant_key",
        "user_tx_count_30d",
        "user_total_amount_30d",
        "user_avg_amount_30d",
        "merchant_tx_count_30d",
        "merchant_unique_users_30d",
        "velocity_1hr",
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
    ]
    feature_df = feature_df[[col for col in keep_cols if col in feature_df.columns]]
    try:
        preprocessed_df = preprocess_data(feature_df, is_train=False)
    except Exception as pre_ex:
        raise HTTPException(
            status_code=500,
            detail=f"Inference preprocessing failure: {pre_ex}",
        )

    # 4. Model Prediction
    try:
        prob = float(MODEL.predict_proba(preprocessed_df)[0, 1])
    except Exception as model_ex:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference execution failure: {model_ex}",
        )

    # Classification Threshold logic
    api_config = config.get("api", {})
    fraud_thresh = float(api_config.get("fraud_threshold", 0.5))
    susp_thresh = float(api_config.get("suspicious_threshold", 0.3))

    if prob >= fraud_thresh:
        prediction = "FRAUD"
        confidence = "HIGH" if prob >= 0.70 else "MEDIUM"
    elif prob >= susp_thresh:
        prediction = "SUSPICIOUS"
        confidence = "MEDIUM"
    else:
        prediction = "CLEAN"
        confidence = "HIGH" if prob < 0.15 else "MEDIUM"

    # 5. SHAP Explanation
    shap_explanation = {}
    if EXPLAINER is not None:
        try:
            shap_values = EXPLAINER.shap_values(preprocessed_df)
            shap_dict = dict(zip(preprocessed_df.columns, shap_values[0]))
            # Keep only values contributing at least 0.0001
            shap_explanation = {
                k: float(v) for k, v in shap_dict.items() if abs(v) > 1e-4
            }
        except Exception as shap_ex:
            print(f"Warning: SHAP explanation generation failed: {shap_ex}")

    response_dict = {
        "transaction_id": payload.transaction_id,
        "fraud_probability": prob,
        "prediction": prediction,
        "confidence": confidence,
        "model_version": MODEL_VERSION,
        "shap_explanation": shap_explanation,
    }

    # 6. Save prediction metadata log
    save_prediction_record(response_dict, db)

    return response_dict


@app.get("/metrics")
def get_metrics(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Retrieve production model classification evaluation metrics."""
    # 1. Try DB
    if db is not None:
        try:
            res = db.execute(
                text(
                    "SELECT f1_score, auc_score, precision_score, recall_score "
                    "FROM model_registry WHERE is_production = True "
                    "ORDER BY id DESC LIMIT 1"
                )
            )
            row = res.fetchone()
            if row:
                return {
                    "f1_score": float(row[0]),
                    "auc_score": float(row[1]),
                    "precision_score": float(row[2]),
                    "recall_score": float(row[3]),
                }
        except Exception:
            pass

    # 2. Fallback to local JSON
    local_metadata_path = Path("models/model_registry_metadata.json")
    if local_metadata_path.exists():
        try:
            with open(local_metadata_path, "r", encoding="utf-8") as f:
                records = json.load(f)
                prod_records = [r for r in records if r.get("is_production") is True]
                if prod_records:
                    last = prod_records[-1]
                    return {
                        "f1_score": float(last["f1_score"]),
                        "auc_score": float(last["auc_score"]),
                        "precision_score": float(last["precision_score"]),
                        "recall_score": float(last["recall_score"]),
                    }
        except Exception:
            pass

    return {
        "f1_score": 0.0,
        "auc_score": 0.0,
        "precision_score": 0.0,
        "recall_score": 0.0,
    }


@app.get("/drift")
def get_drift(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Retrieve the latest statistical data drift metrics summary."""
    # 1. Try DB
    if db is not None:
        try:
            res = (
                db.query(DriftLogRecord)
                .order_by(DriftLogRecord.id.desc())
                .limit(5)
                .all()
            )
            if res:
                return {
                    "run_date": res[0].run_date.isoformat(),
                    "drift_detected": any(r.drift_detected for r in res),
                    "features": [
                        {
                            "name": r.feature_name,
                            "drift_score": r.drift_score,
                            "p_value": r.p_value,
                            "drift_detected": r.drift_detected,
                        }
                        for r in res
                    ],
                }
        except Exception:
            pass

    # 2. Fallback JSON
    drift_file = Path("logs/drift_logs.json")
    if drift_file.exists():
        try:
            with open(drift_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                if records:
                    latest_date = records[-1].get("run_date")
                    latest_runs = [
                        r for r in records if r.get("run_date") == latest_date
                    ]
                    return {
                        "run_date": latest_date,
                        "drift_detected": any(
                            r.get("drift_detected") for r in latest_runs
                        ),
                        "features": [
                            {
                                "name": r.get("feature_name"),
                                "drift_score": r.get("drift_score"),
                                "p_value": r.get("p_value"),
                                "drift_detected": r.get("drift_detected"),
                            }
                            for r in latest_runs
                        ],
                    }
        except Exception:
            pass

    return {"drift_detected": False, "features": []}


@app.get("/drift/history")
def get_drift_history(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """Retrieve historical statistical drift runs."""
    history = []
    # 1. Try DB
    if db is not None:
        try:
            records = (
                db.query(DriftLogRecord)
                .order_by(DriftLogRecord.id.desc())
                .limit(100)
                .all()
            )
            for r in records:
                history.append(
                    {
                        "run_date": r.run_date.isoformat(),
                        "feature_name": r.feature_name,
                        "drift_score": r.drift_score,
                        "p_value": r.p_value,
                        "drift_detected": r.drift_detected,
                        "drift_intensity": r.drift_intensity,
                    }
                )
            return history
        except Exception:
            pass

    # 2. Fallback JSON
    drift_file = Path("logs/drift_logs.json")
    if drift_file.exists():
        try:
            with open(drift_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    return history


@app.post("/retrain")
def trigger_manual_retrain(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Manually trigger model retraining in an asynchronous background thread."""
    background_tasks.add_task(retrain_pipeline)
    return {"status": "retraining started"}


@app.get("/model/info")
def get_model_info(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Retrieve metadata information and promotion logs for the active model."""
    promotion_history = []

    # 1. Try DB
    if db is not None:
        try:
            res = (
                db.query(RetrainLogRecord)
                .order_by(RetrainLogRecord.id.desc())
                .limit(10)
                .all()
            )
            for r in res:
                promotion_history.append(
                    {
                        "triggered_at": r.triggered_at.isoformat(),
                        "trigger_reason": r.trigger_reason,
                        "old_f1": r.old_f1,
                        "new_f1": r.new_f1,
                        "promoted": r.promoted,
                        "notes": r.notes,
                    }
                )
        except Exception:
            pass

    # 2. Fallback JSON
    if not promotion_history:
        history_file = Path("logs/retraining_history.json")
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    for r in history:
                        promotion_history.append(
                            {
                                "triggered_at": r.get("triggered_at"),
                                "trigger_reason": r.get("trigger_reason"),
                                "old_f1": r.get("old_f1"),
                                "new_f1": r.get("new_f1"),
                                "promoted": r.get("promoted"),
                                "notes": r.get("notes"),
                            }
                        )
            except Exception:
                pass

    return {
        "model_version": MODEL_VERSION,
        "model_source": MODEL_SOURCE,
        "promotion_history": promotion_history,
    }


@app.get("/transactions/recent")
def get_recent_transactions(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """Retrieve the last 100 logged prediction results."""
    recent = []

    # 1. Try DB
    if db is not None:
        try:
            res = (
                db.query(PredictionRecord)
                .order_by(PredictionRecord.id.desc())
                .limit(100)
                .all()
            )
            for r in res:
                recent.append(
                    {
                        "transaction_id": r.transaction_id,
                        "fraud_probability": r.fraud_probability,
                        "prediction": r.prediction,
                        "model_version": r.model_version,
                        "shap_explanation": r.shap_values,
                        "timestamp": r.timestamp.isoformat(),
                    }
                )
            return recent
        except Exception:
            pass

    # 2. Fallback JSON
    predictions_file = Path("logs/predictions.json")
    if predictions_file.exists():
        try:
            with open(predictions_file, "r", encoding="utf-8") as f:
                all_preds = json.load(f)
                recent = all_preds[-100:]
                # Reverse to match descending order
                recent.reverse()
        except Exception:
            pass

    return recent


@app.get("/grafana-metrics")
def get_grafana_metrics(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """Serve formatted time-series metrics for Grafana Cloud monitoring."""
    recent_preds = get_recent_transactions(db)
    total_preds = len(recent_preds)

    fraud_count = sum(1 for p in recent_preds if p.get("prediction") == "FRAUD")
    fraud_rate = (fraud_count / total_preds) if total_preds > 0 else 0.0

    latest_drift = get_drift(db)
    drift_detected = 1 if latest_drift.get("drift_detected") else 0

    timestamp_ms = int(time.time() * 1000)
    avg_latency = (
        sum(LATENCY_HISTORY) / len(LATENCY_HISTORY) if LATENCY_HISTORY else 0.0
    )

    return [
        {
            "target": "fraud_rate",
            "datapoints": [[fraud_rate, timestamp_ms]],
        },
        {
            "target": "total_predictions",
            "datapoints": [[float(total_preds), timestamp_ms]],
        },
        {
            "target": "drift_detected",
            "datapoints": [[float(drift_detected), timestamp_ms]],
        },
        {
            "target": "avg_latency_ms",
            "datapoints": [[avg_latency, timestamp_ms]],
        },
    ]


@app.get("/health")
def health() -> dict[str, str]:
    """Check API server, database connectivity, and production model health."""
    db_status = "connected" if check_db_health() else "offline"
    model_status = "loaded" if MODEL is not None else "missing"

    return {
        "status": "ok" if MODEL is not None else "degraded",
        "database": db_status,
        "model": model_status,
        "model_source": MODEL_SOURCE,
    }
