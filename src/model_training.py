"""Model training entry points and helpers."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy.sql import func
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from src.data_ingestion import load_config


def setup_mlflow(config: dict[str, Any]) -> None:
    """Configure MLflow tracking URI.

    Falls back to local file storage if server is down.
    """
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    # Try connecting to the tracking server
    try:
        with urllib.request.urlopen(tracking_uri, timeout=1.5):
            mlflow.set_tracking_uri(tracking_uri)
            print(f"MLflow tracking URI set to server: {tracking_uri}")
    except Exception:
        mlflow.set_tracking_uri("file:./mlruns")
        print(
            "MLflow server not reachable. "
            "Falling back to local './mlruns' directory."
        )

    mlflow.set_experiment("fraudshield-classifier")


def log_model_metadata(
    model_name: str,
    model_version: int | str,
    run_id: str,
    f1_score: float,
    auc_score: float,
    precision_score: float,
    recall_score: float,
    threshold: float,
    is_production: bool = True,
    database_url: str | None = None,
) -> None:
    """Log model metadata to PostgreSQL with local JSON file fallback."""
    metadata_dict = {
        "model_name": model_name,
        "model_version": str(model_version),
        "run_id": run_id,
        "f1_score": float(f1_score),
        "auc_score": float(auc_score),
        "precision_score": float(precision_score),
        "recall_score": float(recall_score),
        "threshold": float(threshold),
        "is_production": bool(is_production),
        "created_at": pd.Timestamp.now().isoformat(),
    }

    db_url = database_url or os.getenv("DATABASE_URL")
    if db_url:
        try:
            engine = create_engine(db_url)
            meta = MetaData()
            model_registry = Table(
                "model_registry",
                meta,
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
            )
            meta.create_all(engine)
            with engine.connect() as conn:
                conn.execute(
                    model_registry.insert().values(
                        model_name=metadata_dict["model_name"],
                        model_version=metadata_dict["model_version"],
                        run_id=metadata_dict["run_id"],
                        f1_score=metadata_dict["f1_score"],
                        auc_score=metadata_dict["auc_score"],
                        precision_score=metadata_dict["precision_score"],
                        recall_score=metadata_dict["recall_score"],
                        threshold=metadata_dict["threshold"],
                        is_production=metadata_dict["is_production"],
                    )
                )
                conn.commit()
            print(
                "Successfully logged model metadata to PostgreSQL "
                "model_registry table."
            )
            return
        except Exception as e:
            print(f"Warning: Failed to log to PostgreSQL model_registry table: {e}")
            print("Falling back to local JSON file logging.")

    # Local JSON fallback
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    registry_file = models_dir / "model_registry_metadata.json"

    records = []
    if registry_file.exists():
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                if not isinstance(records, list):
                    records = []
        except Exception:
            records = []

    records.append(metadata_dict)
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
    print(f"Logged model metadata locally to {registry_file}.")


def train_model(
    config_path: str | Path = "config/config.yaml",
    database_url: str | None = None,
    promote: bool = True,
) -> tuple[Any, str]:
    """Train XGBoost model, tune hyperparams, and log to MLflow/registry."""
    config = load_config(config_path)
    paths_config = config["paths"]
    model_config = config["model"]
    project_random_state = config["project"]["random_state"]

    processed_dir = Path(paths_config["processed_data_dir"])
    models_dir = Path(paths_config["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Processed splits
    X_train = pd.read_csv(processed_dir / "train_feats.csv")
    y_train = pd.read_csv(processed_dir / "train_target.csv").iloc[:, 0]
    X_val = pd.read_csv(processed_dir / "val_feats.csv")
    y_val = pd.read_csv(processed_dir / "val_target.csv").iloc[:, 0]

    # 2. Setup MLflow
    setup_mlflow(config)

    # 3. Define tuning search space
    # 3 options: default, shallow trees, higher learning rate
    tuning_space = [
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 100, "subsample": 0.8},
        # Default config
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.8},
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 200, "subsample": 1.0},
    ]

    best_model = None
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_params = {}
    best_run_id = ""

    # Enable autologging
    mlflow.xgboost.autolog(log_models=True)

    print("Starting hyperparameter tuning...")
    with mlflow.start_run(run_name="Hyperparameter_Tuning"):
        for i, params in enumerate(tuning_space):
            run_name = f"tuning_run_{i}"
            with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                # Calculate scale_pos_weight dynamically to handle any class imbalances
                pos_count = (y_train == 1).sum()
                neg_count = (y_train == 0).sum()
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

                model = XGBClassifier(
                    objective=model_config["xgboost"]["objective"],
                    eval_metric=model_config["xgboost"]["eval_metric"],
                    colsample_bytree=model_config["xgboost"]["colsample_bytree"],
                    random_state=project_random_state,
                    scale_pos_weight=scale_pos_weight,
                    **params,
                )

                model.fit(X_train, y_train)

                # Evaluate on validation set with default 0.5 threshold
                val_preds = model.predict(X_val)
                # Calculate validation metrics manually to log

                val_f1 = float(f1_score(y_val, val_preds))
                val_prec = float(precision_score(y_val, val_preds))
                val_rec = float(recall_score(y_val, val_preds))

                mlflow.log_metric("val_f1", val_f1)
                mlflow.log_metric("val_precision", val_prec)
                mlflow.log_metric("val_recall", val_rec)

                print(f"Trial {i} {params} -> Val F1: {val_f1:.4f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_prec = val_prec
                    best_rec = val_rec
                    best_model = model
                    best_params = params
                    best_run_id = child_run.info.run_id

        # Log best params to parent run
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_f1", best_f1)

    print(
        f"Hyperparameter tuning complete. Best Trial F1: {best_f1:.4f} "
        f"with {best_params}"
    )

    # Register model in MLflow registry and tag as Production
    model_name = os.getenv("MODEL_NAME", "fraudshield-xgboost")
    try:
        model_uri = f"runs:/{best_run_id}/model"
        model_details = mlflow.register_model(model_uri, model_name)
        client = mlflow.tracking.MlflowClient()
        if promote:
            client.set_registered_model_alias(
                model_name, "Production", model_details.version
            )
            print(
                f"Registered model '{model_name}' version "
                f"{model_details.version} and tagged as 'Production'."
            )
        else:
            print(
                f"Registered model '{model_name}' version "
                f"{model_details.version} (not promoted)."
            )
        model_version = model_details.version
    except Exception as e:
        print(f"Warning: Could not register model in MLflow Registry: {e}")
        model_version = "1"

    # Log best model metadata to postgres registry / local fallback JSON
    log_model_metadata(
        model_name=model_name,
        model_version=model_version,
        run_id=best_run_id,
        f1_score=best_f1,
        # AUC is set to 0.0 here; the real value is written by evaluate_model()
        auc_score=0.0,
        precision_score=best_prec,
        recall_score=best_rec,
        threshold=0.5,
        is_production=promote,
        database_url=database_url,
    )

    # Save best model locally
    joblib.dump(best_model, models_dir / "xgboost_model.joblib")
    print(f"Saved best model locally to {models_dir / 'xgboost_model.joblib'}")

    return best_model, best_run_id
