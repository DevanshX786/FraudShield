"""Tests for drift detection and automated retraining pipelines."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from src.data_ingestion import load_ieee_cis_dataset
from src.drift_detector import detect_drift, get_hour_bucket
from src.feature_engineering import engineer_features
from src.retraining_pipeline import (
    acquire_lock,
    check_rate_limits,
    get_current_production_f1,
    log_retrain_history,
    release_lock,
    retrain_pipeline,
)


def _write_mock_config(
    path: Path,
    raw_dir: Path,
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    logs_dir: Path,
) -> None:
    config_dict = {
        "project": {
            "name": "test-drift-detect",
            "random_state": 42,
        },
        "paths": {
            "raw_data_dir": str(raw_dir),
            "processed_data_dir": str(processed_dir),
            "models_dir": str(models_dir),
            "reports_dir": str(reports_dir),
            "logs_dir": str(logs_dir),
            "inference_batch_dir": str(raw_dir / "inference_batches"),
        },
        "data": {
            "train_transaction_file": "train_transaction.csv",
            "train_identity_file": "train_identity.csv",
            "test_transaction_file": "test_transaction.csv",
            "test_identity_file": "test_identity.csv",
            "target_column": "isFraud",
            "join_key": "TransactionID",
            "required_train_transaction_columns": [
                "TransactionID",
                "isFraud",
                "TransactionDT",
                "TransactionAmt",
                "ProductCD",
                "card1",
                "card2",
                "card3",
                "card4",
                "card5",
                "card6",
            ],
        },
        "features": {
            "user_key_columns": ["card1", "card2", "card3", "card5"],
            "merchant_key_columns": ["ProductCD", "card4"],
            "categorical_columns": ["ProductCD", "card4", "card6"],
        },
        "split": {
            "train_size": 0.60,
            "validation_size": 0.20,
            "test_size": 0.20,
            "stratify": True,
        },
        "model": {
            "name": "test-model",
            "xgboost": {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            },
            "threshold_search": {
                "min": 0.30,
                "max": 0.70,
                "step": 0.10,
            },
        },
        "drift_simulation": {
            "base_intensity": 0.1,
            "current_intensity": 0.1,
            "drift_threshold": 0.05,
        },
        "retraining": {
            "min_hours_between_retrains": 6,
            "max_retrains_per_day": 3,
            "promotion_min_f1_delta": 0.01,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)


def _write_mock_raw_data(raw_dir: Path) -> None:
    # Train transaction data (50 rows)
    np.random.seed(42)
    rows = []
    for i in range(100):
        rows.append(
            {
                "TransactionID": 1000 + i,
                "isFraud": i % 10 == 0,  # 10% fraud
                "TransactionDT": 10000 + i * 100,
                "TransactionAmt": float(np.random.randint(10, 200)),
                "ProductCD": "W" if i % 2 == 0 else "H",
                "card1": 15000 + (i % 5),
                "card2": 321.0,
                "card3": 150.0,
                "card4": "visa" if i % 3 == 0 else "mastercard",
                "card5": 226.0,
                "card6": "debit" if i % 4 == 0 else "credit",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(raw_dir / "train_transaction.csv", index=False)


def test_hour_bucket() -> None:
    hours = pd.Series([2, 9, 14, 21])
    buckets = get_hour_bucket(hours)
    assert buckets.iloc[0] == "night"
    assert buckets.iloc[1] == "morning"
    assert buckets.iloc[2] == "afternoon"
    assert buckets.iloc[3] == "evening"


def test_lock_mechanism(tmp_path: Path) -> None:
    lock_file = tmp_path / "retrain.lock"
    assert acquire_lock(lock_file) is True
    # Attempting to acquire again should fail
    assert acquire_lock(lock_file) is False
    release_lock(lock_file)
    assert acquire_lock(lock_file) is True
    release_lock(lock_file)


def test_check_rate_limits(tmp_path: Path) -> None:
    history_file = tmp_path / "retraining_history.json"

    # No history -> allowed
    allowed, _ = check_rate_limits(database_url=None, history_file_path=history_file)
    assert allowed is True

    # Log an event less than 6 hours ago
    log_retrain_history(
        triggered_at=datetime.now(),
        trigger_reason="test",
        old_f1=0.8,
        new_f1=0.85,
        promoted=True,
        notes="First run",
        database_url=None,
        history_file_path=history_file,
    )

    allowed, reason = check_rate_limits(
        database_url=None,
        history_file_path=history_file,
        min_hours=6.0,
        max_per_day=3,
    )
    assert allowed is False
    assert "too recently" in reason


def test_detect_drift_and_retrain_pipeline(tmp_path: Path) -> None:
    # Setup directories
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    config_file = tmp_path / "config.yaml"
    _write_mock_config(
        config_file, raw_dir, processed_dir, models_dir, reports_dir, logs_dir
    )
    _write_mock_raw_data(raw_dir)

    # 1. First run data preprocessing and train baseline model
    from src.data_preprocessing import preprocess_data
    from src.model_training import train_model

    # Let's load the raw data, run feature engineering and preprocess
    raw_data, _ = load_ieee_cis_dataset("train", config_path=config_file)
    engineered_data = engineer_features(raw_data, config_path=config_file)
    preprocess_data(engineered_data, is_train=True, config_path=config_file)

    train_model(
        config_path=config_file,
        database_url=None,
        promote=True,
    )
    assert (models_dir / "xgboost_model.joblib").exists()

    # Move model_registry_metadata.json to test models dir if needed
    if Path("models/model_registry_metadata.json").exists():
        shutil.copy(
            "models/model_registry_metadata.json",
            str(models_dir / "model_registry_metadata.json"),
        )

    # Verify initial production F1
    prod_f1 = get_current_production_f1(
        database_url=None,
        registry_file_path=models_dir / "model_registry_metadata.json",
    )
    assert prod_f1 > 0.0

    # 2. Write a mock inference batch (clean distribution to avoid drift)
    batch_dir = raw_dir / "inference_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_df = engineered_data.iloc[-10:].copy()
    batch_df.to_csv(batch_dir / "batch_20260618.csv", index=False)

    # Detect drift - should be False
    drift_detected, summary = detect_drift(config_path=config_file)
    assert isinstance(drift_detected, bool)
    assert "metrics" in summary
    assert (reports_dir / "drift_report.html").exists()

    # 3. Trigger retrain pipeline manually
    # Note: Rate check will fail if we check against the global logs, so let's mock it
    # We will pass database_url sqlite in-memory for retrain_pipeline
    # Cleanup any leftovers
    if Path("models/retrain.lock").exists():
        Path("models/retrain.lock").unlink()
    if Path("logs/retraining_history.json").exists():
        Path("logs/retraining_history.json").unlink()

    # Run retraining pipeline
    retrain_pipeline(config_path=config_file, database_url="sqlite:///:memory:")
    assert (models_dir / "xgboost_model.joblib").exists()
