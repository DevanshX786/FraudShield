"""Tests for model training and evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.model_evaluation import evaluate_model
from src.model_training import train_model


def _write_mock_config(
    path: Path, processed_dir: Path, models_dir: Path, reports_dir: Path
) -> None:
    config_dict = {
        "project": {
            "name": "test-project",
            "random_state": 42,
        },
        "paths": {
            "processed_data_dir": str(processed_dir),
            "models_dir": str(models_dir),
            "reports_dir": str(reports_dir),
        },
        "data": {
            "target_column": "isFraud",
            "join_key": "TransactionID",
        },
        "model": {
            "xgboost": {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "colsample_bytree": 1.0,
            },
            "threshold_search": {
                "min": 0.30,
                "max": 0.70,
                "step": 0.10,
            },
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)


def test_train_and_evaluate_pipeline(tmp_path: Path) -> None:
    # Set up directory paths
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    config_file = tmp_path / "config.yaml"
    _write_mock_config(config_file, processed_dir, models_dir, reports_dir)

    # Create mock processed dataset splits
    np.random.seed(42)
    n_features = 5
    n_train = 30
    n_val = 10
    n_test = 10

    # Features
    cols = [f"feat_{i}" for i in range(n_features)]
    train_feats = pd.DataFrame(np.random.randn(n_train, n_features), columns=cols)
    val_feats = pd.DataFrame(np.random.randn(n_val, n_features), columns=cols)
    test_feats = pd.DataFrame(np.random.randn(n_test, n_features), columns=cols)

    # Target (balanced clean/fraud for simple training)
    train_target = pd.DataFrame({"isFraud": [0, 1] * 15})
    val_target = pd.DataFrame({"isFraud": [0, 1] * 5})
    test_target = pd.DataFrame({"isFraud": [0, 1] * 5})

    # Save to mock processed directory
    train_feats.to_csv(processed_dir / "train_feats.csv", index=False)
    train_target.to_csv(processed_dir / "train_target.csv", index=False)
    val_feats.to_csv(processed_dir / "val_feats.csv", index=False)
    val_target.to_csv(processed_dir / "val_target.csv", index=False)
    test_feats.to_csv(processed_dir / "test_feats.csv", index=False)
    test_target.to_csv(processed_dir / "test_target.csv", index=False)

    # 1. Test train_model (local file storage and SQLite DB)
    model, run_id = train_model(
        config_path=config_file, database_url="sqlite:///:memory:"
    )

    assert model is not None
    assert isinstance(run_id, str)
    assert len(run_id) > 0
    assert (models_dir / "xgboost_model.joblib").exists()

    # 2. Test evaluate_model
    report = evaluate_model(model, config_path=config_file)

    assert "metrics" in report
    assert "global_shap_importance" in report
    assert "local_explanations_sample" in report

    metrics = report["metrics"]
    assert "f1_score" in metrics
    assert "precision_score" in metrics
    assert "recall_score" in metrics
    assert "roc_auc_score" in metrics
    assert "threshold" in metrics
    assert "confusion_matrix" in metrics

    # Verify report is written to disk
    assert (reports_dir / "evaluation_report.json").exists()
    with open(reports_dir / "evaluation_report.json", "r", encoding="utf-8") as f:
        loaded_report = json.load(f)
        assert "metrics" in loaded_report
