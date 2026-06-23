"""Automated model retraining pipeline with statistical validation and safety checks."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
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
    text,
)
from sqlalchemy.sql import func

from src.data_ingestion import load_config, load_ieee_cis_dataset
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.model_evaluation import evaluate_model
from src.model_training import setup_mlflow, train_model


def acquire_lock(lock_path: Path) -> bool:
    """Try to acquire an exclusive retraining file lock."""
    try:
        with open(lock_path, "x") as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False


def release_lock(lock_path: Path) -> None:
    """Release the retraining file lock."""
    if lock_path.exists():
        try:
            lock_path.unlink()
        except Exception as e:
            print(f"Warning: Failed to delete lock file {lock_path}: {e}")


def check_rate_limits(
    database_url: str | None = None,
    history_file_path: Path = Path("logs/retraining_history.json"),
    min_hours: float = 6.0,
    max_per_day: int = 3,
) -> tuple[bool, str]:
    """Check if retraining is allowed based on hourly gap and daily count limit."""
    db_url = database_url or os.getenv("DATABASE_URL")
    runs = []

    # 1. Try reading from PostgreSQL
    if db_url:
        try:
            engine = create_engine(db_url)
            with engine.connect() as conn:
                res = conn.execute(
                    text(
                        "SELECT timestamp FROM retrain_logs "
                        "ORDER BY id DESC LIMIT 10"
                    )
                )
                for row in res.fetchall():
                    val = row[0]
                    if isinstance(val, str):
                        val = pd.to_datetime(val).to_pydatetime()
                    runs.append(val)
        except Exception as e:
            print(f"Warning: Could not query retrain_logs database table: {e}")

    # 2. Fallback to local JSON history file
    if not runs and history_file_path.exists():
        try:
            with open(history_file_path, "r", encoding="utf-8") as f:
                history = json.load(f)
                if isinstance(history, list):
                    for record in history:
                        ts_str = record.get("timestamp") or record.get("triggered_at")
                        if ts_str:
                            runs.append(datetime.fromisoformat(ts_str))
        except Exception as e:
            print(f"Warning: Could not read retraining history JSON: {e}")

    if not runs:
        return True, ""

    # Sort runs ascending
    runs = sorted(runs)
    now = datetime.now()

    # Gap check (min hours)
    last_run = runs[-1]
    if now - last_run < timedelta(hours=min_hours):
        time_left = timedelta(hours=min_hours) - (now - last_run)
        return (
            False,
            f"Retraining occurred too recently (last: {last_run}). "
            f"Please wait {time_left.total_seconds() / 3600:.2f} hours.",
        )

    # Count check (max per day)
    cutoff = now - timedelta(days=1)
    recent_runs = [r for r in runs if r >= cutoff]
    if len(recent_runs) >= max_per_day:
        return (
            False,
            f"Daily retraining limit reached ({len(recent_runs)} "
            f"runs in the last 24h). Max allowed: {max_per_day}.",
        )

    return True, ""


def get_current_production_f1(
    database_url: str | None = None,
    registry_file_path: Path = Path("models/model_registry_metadata.json"),
) -> float:
    """Retrieve the F1 score of the current Production model."""
    db_url = database_url or os.getenv("DATABASE_URL")

    # 1. Try reading from PostgreSQL
    if db_url:
        try:
            engine = create_engine(db_url)
            with engine.connect() as conn:
                res = conn.execute(
                    text(
                        "SELECT f1_score FROM model_registry "
                        "WHERE is_production = :is_prod "
                        "ORDER BY id DESC LIMIT 1"
                    ),
                    {"is_prod": True},
                )
                row = res.fetchone()
                if row:
                    return float(row[0])
        except Exception as e:
            print(f"Warning: Failed to fetch production F1 from database: {e}")

    # 2. Fallback to local JSON registry metadata
    if registry_file_path.exists():
        try:
            with open(registry_file_path, "r", encoding="utf-8") as f:
                records = json.load(f)
                prod_records = [r for r in records if r.get("is_production") is True]
                if prod_records:
                    return float(prod_records[-1]["f1_score"])
        except Exception as e:
            print(f"Warning: Failed to read local JSON registry F1: {e}")

    return 0.0


def promote_model_metadata(
    run_id: str,
    database_url: str | None = None,
    registry_file_path: Path = Path("models/model_registry_metadata.json"),
) -> None:
    """Promote the model with run_id to production in database and JSON."""
    db_url = database_url or os.getenv("DATABASE_URL")
    model_name = os.getenv("MODEL_NAME", "fraudshield-xgboost")

    # 1. Update PostgreSQL
    if db_url:
        try:
            engine = create_engine(db_url)
            with engine.connect() as conn:
                # De-promote existing models
                conn.execute(
                    text(
                        "UPDATE model_registry SET is_production = False "
                        "WHERE model_name = :name"
                    ),
                    {"name": model_name},
                )
                # Promote the new model
                conn.execute(
                    text(
                        "UPDATE model_registry SET is_production = True "
                        "WHERE run_id = :run_id"
                    ),
                    {"run_id": run_id},
                )
                conn.commit()
            print("Successfully updated model registry database promotion state.")
        except Exception as e:
            print(f"Warning: Failed to update database promotion state: {e}")

    # 2. Update local JSON registry metadata
    if registry_file_path.exists():
        try:
            with open(registry_file_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if isinstance(records, list):
                for r in records:
                    if r.get("model_name") == model_name:
                        r["is_production"] = r.get("run_id") == run_id
                with open(registry_file_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=4)
                print("Successfully updated local model registry promotion state.")
        except Exception as e:
            print(f"Warning: Failed to update local model registry JSON: {e}")


def log_retrain_history(
    triggered_at: datetime,
    trigger_reason: str,
    old_f1: float,
    new_f1: float,
    promoted: bool,
    notes: str,
    database_url: str | None = None,
    history_file_path: Path | None = None,
) -> None:
    """Log retraining attempt metadata to database and JSON."""
    db_url = database_url or os.getenv("DATABASE_URL")
    timestamp_val = datetime.now()

    # 1. Update PostgreSQL
    if db_url:
        try:
            engine = create_engine(db_url)
            meta = MetaData()
            retrain_logs = Table(
                "retrain_logs",
                meta,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("triggered_at", DateTime),
                Column("trigger_reason", String),
                Column("old_f1", Float),
                Column("new_f1", Float),
                Column("promoted", Boolean),
                Column("notes", String),
                Column("timestamp", DateTime, default=func.now()),
            )
            meta.create_all(engine)
            with engine.connect() as conn:
                conn.execute(
                    retrain_logs.insert().values(
                        triggered_at=triggered_at,
                        trigger_reason=trigger_reason,
                        old_f1=old_f1,
                        new_f1=new_f1,
                        promoted=promoted,
                        notes=notes,
                    )
                )
                conn.commit()
            print("Successfully logged retraining event to PostgreSQL.")
        except Exception as e:
            print(f"Warning: Failed to log retraining to database: {e}")

    # 2. Update local JSON history file
    if history_file_path:
        history_file = Path(history_file_path)
    else:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        history_file = logs_dir / "retraining_history.json"

    records = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                if not isinstance(records, list):
                    records = []
        except Exception:
            records = []

    records.append(
        {
            "triggered_at": triggered_at.isoformat(),
            "trigger_reason": trigger_reason,
            "old_f1": float(old_f1),
            "new_f1": float(new_f1),
            "promoted": bool(promoted),
            "notes": notes,
            "timestamp": timestamp_val.isoformat(),
        }
    )

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
    print(f"Successfully logged retraining event to {history_file}.")


def reset_drift_intensity(config_path: str | Path) -> None:
    """Reset the current simulation drift intensity back to baseline."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if "drift_simulation" in config:
            base_intensity = config["drift_simulation"].get("base_intensity", 0.1)
            config["drift_simulation"]["current_intensity"] = base_intensity

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f)
            print(
                "Reset current drift intensity in config "
                f"to baseline: {base_intensity}"
            )
    except Exception as e:
        print(f"Warning: Failed to reset drift intensity in config: {e}")


def retrain_pipeline(
    config_path: str | Path = "config/config.yaml",
    database_url: str | None = None,
) -> None:
    """Execute automated retraining workflow, validate, and promote."""
    start_time = datetime.now()
    config = load_config(config_path)
    paths_config = config["paths"]
    retrain_config = config["retraining"]

    lock_file_path = Path(paths_config["models_dir"]) / "retrain.lock"
    history_file_path = Path(paths_config["logs_dir"]) / "retraining_history.json"

    # 1. Acquire Concurrency Lock
    print("Attempting to acquire retraining lock...")
    if not acquire_lock(lock_file_path):
        print("Retraining aborting: Lock file already exists.", file=sys.stderr)
        return

    try:
        # 2. Check Rate Limits
        print("Checking retraining rate limits...")
        min_hours = float(retrain_config["min_hours_between_retrains"])
        max_per_day = int(retrain_config["max_retrains_per_day"])

        allowed, reason = check_rate_limits(
            database_url=database_url,
            history_file_path=history_file_path,
            min_hours=min_hours,
            max_per_day=max_per_day,
        )
        if not allowed:
            print(f"Retraining rate limit hit: {reason}", file=sys.stderr)
            # Log failure event
            log_retrain_history(
                triggered_at=start_time,
                trigger_reason="drift_detection",
                old_f1=0.0,
                new_f1=0.0,
                promoted=False,
                notes=f"Rate limit check failed: {reason}",
                database_url=database_url,
            )
            return

        # 3. Load baseline and inference batch files
        print("Loading raw training baseline data...")
        raw_train_df, _ = load_ieee_cis_dataset("train", config_path=config_path)

        batch_dir = Path(paths_config["inference_batch_dir"])
        batch_files = list(batch_dir.glob("batch_*.csv"))

        if batch_files:
            print(f"Found {len(batch_files)} inference batches. Concatenating...")
            batch_dfs = []
            for bf in batch_files:
                b_df = pd.read_csv(bf)
                # Align columns and map 'amount' to 'TransactionAmt' if necessary
                if "amount" in b_df.columns and "TransactionAmt" not in b_df.columns:
                    b_df = b_df.rename(columns={"amount": "TransactionAmt"})
                # Filter to match columns in raw training dataframe
                b_df = b_df[[c for c in raw_train_df.columns if c in b_df.columns]]
                batch_dfs.append(b_df)

            combined_raw = pd.concat(
                [raw_train_df] + batch_dfs, axis=0, ignore_index=True
            )
            # Sort chronologically to preserve feature engineering correct state
            combined_raw = combined_raw.sort_values("TransactionDT").reset_index(
                drop=True
            )
        else:
            print("No inference batches found. Retraining on raw baseline only.")
            combined_raw = raw_train_df.copy()

        # 4. Feature Engineering
        print("Re-running feature engineering on combined dataset...")
        engineered_df = engineer_features(combined_raw, config_path=config_path)

        # 5. Data Preprocessing (fits new imputer, scaler, OHE, and SMOTE)
        # Drop rows where isFraud is NaN - can happen when inference batch rows
        # are concatenated and they lack the label column.
        target_col = config["data"]["target_column"]
        if target_col in engineered_df.columns:
            before = len(engineered_df)
            engineered_df = engineered_df.dropna(subset=[target_col]).copy()
            dropped = before - len(engineered_df)
            if dropped > 0:
                print(
                    f"Dropped {dropped} rows with missing '{target_col}' "
                    "labels before preprocessing."
                )
        print("Preprocessing combined data splits...")
        preprocess_data(engineered_df, is_train=True, config_path=config_path)

        # 6. Train model (promote=False so we don't automatically set alias Production)
        print("Training candidate model on updated dataset splits...")
        candidate_model, new_run_id = train_model(
            config_path=config_path,
            database_url=database_url,
            promote=False,
        )

        # 7. Evaluate model (threshold selection and final metrics)
        print("Evaluating candidate model performance...")
        report = evaluate_model(candidate_model, config_path=config_path)
        new_f1 = float(report["metrics"]["f1_score"])

        # 8. Retrieve Production Model F1
        old_f1 = get_current_production_f1(
            database_url=database_url,
            registry_file_path=Path(paths_config["models_dir"])
            / "model_registry_metadata.json",
        )

        print(f"Candidate F1: {new_f1:.4f} vs Current Production F1: {old_f1:.4f}")
        delta = new_f1 - old_f1
        min_delta = float(retrain_config["promotion_min_f1_delta"])

        # 9. Promotion Check
        if delta >= min_delta or old_f1 == 0.0:
            print(
                f"Promotion criteria met! Improvement of {delta:.4f} "
                f">= min delta of {min_delta:.4f}."
            )
            # Tag as Production in MLflow
            try:
                setup_mlflow(config)
                import mlflow

                client = mlflow.tracking.MlflowClient()
                model_name = os.getenv("MODEL_NAME", "fraudshield-xgboost")
                versions = client.search_model_versions(f"name='{model_name}'")
                new_version = "1"
                for v in versions:
                    if v.run_id == new_run_id:
                        new_version = v.version
                        break

                client.set_registered_model_alias(model_name, "Production", new_version)
                print(f"MLflow Production alias set to version {new_version}.")
            except Exception as e:
                print(f"Warning: MLflow client failed to set Production alias: {e}")

            # Promote in local/DB records
            promote_model_metadata(
                run_id=new_run_id,
                database_url=database_url,
                registry_file_path=Path(paths_config["models_dir"])
                / "model_registry_metadata.json",
            )

            # Reset simulation drift intensity
            reset_drift_intensity(config_path)

            promoted = True
            notes = (
                f"Model promoted (candidate F1: {new_f1:.4f} > "
                f"production F1: {old_f1:.4f})."
            )
        else:
            print(
                f"Promotion criteria NOT met. Improvement of {delta:.4f} "
                f"is less than {min_delta:.4f}."
            )
            promoted = False
            notes = (
                f"Model NOT promoted (candidate F1: {new_f1:.4f} <= "
                f"production F1: {old_f1:.4f})."
            )

        # Log retraining history
        log_retrain_history(
            triggered_at=start_time,
            trigger_reason="drift_detection",
            old_f1=old_f1,
            new_f1=new_f1,
            promoted=promoted,
            notes=notes,
            database_url=database_url,
        )

    finally:
        # 10. Clean lock file
        print("Releasing retraining lock...")
        release_lock(lock_file_path)


if __name__ == "__main__":
    retrain_pipeline()
