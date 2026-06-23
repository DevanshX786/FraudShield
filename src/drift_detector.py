"""Drift detection using statistical tests and Evidently AI reports."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy.sql import func

from src.data_ingestion import load_config, load_ieee_cis_dataset
from src.feature_engineering import engineer_features


def get_hour_bucket(hour_series: pd.Series) -> pd.Series:
    """Bucket hour (0-23) into night, morning, afternoon, evening."""
    bins = [-1, 6, 12, 18, 24]
    labels = ["night", "morning", "afternoon", "evening"]
    return pd.cut(hour_series, bins=bins, labels=labels).astype(str)


def log_drift_metrics(
    metrics_list: list[dict[str, Any]],
    database_url: str | None = None,
) -> None:
    """Log drift metrics to PostgreSQL with local JSON file fallback."""
    db_url = database_url or os.getenv("DATABASE_URL")
    if db_url:
        try:
            engine = create_engine(db_url)
            meta = MetaData()
            drift_logs = Table(
                "drift_logs",
                meta,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("run_date", Date),
                Column("feature_name", String),
                Column("drift_score", Float),
                Column("p_value", Float),
                Column("drift_detected", Boolean),
                Column("drift_intensity", Float),
                Column("timestamp", DateTime, default=func.now()),
            )
            meta.create_all(engine)
            with engine.connect() as conn:
                for m in metrics_list:
                    # Handle date conversions
                    run_d = m["run_date"]
                    if isinstance(run_d, str):
                        run_d = date.fromisoformat(run_d)

                    conn.execute(
                        drift_logs.insert().values(
                            run_date=run_d,
                            feature_name=m["feature_name"],
                            drift_score=m["drift_score"],
                            p_value=m["p_value"],
                            drift_detected=m["drift_detected"],
                            drift_intensity=m["drift_intensity"],
                        )
                    )
                conn.commit()
            print("Successfully logged drift metrics to PostgreSQL.")
            return
        except Exception as e:
            print(f"Warning: Failed to log to PostgreSQL drift_logs table: {e}")
            print("Falling back to local JSON file logging.")

    # Local JSON fallback
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    drift_logs_file = logs_dir / "drift_logs.json"

    records = []
    if drift_logs_file.exists():
        try:
            with open(drift_logs_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                if not isinstance(records, list):
                    records = []
        except Exception:
            records = []

    # Convert date to string for JSON serialization
    serialized_metrics = []
    for m in metrics_list:
        m_copy = m.copy()
        if isinstance(m_copy["run_date"], date):
            m_copy["run_date"] = m_copy["run_date"].isoformat()
        serialized_metrics.append(m_copy)

    records.extend(serialized_metrics)
    with open(drift_logs_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
    print(f"Logged drift metrics locally to {drift_logs_file}.")


def detect_drift(
    config_path: str | Path = "config/config.yaml",
    database_url: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Compare latest inference batch with baseline training data for drift."""
    config = load_config(config_path)
    paths_config = config["paths"]
    drift_config = config["drift_simulation"]
    drift_threshold = float(drift_config["drift_threshold"])
    current_intensity = float(drift_config["current_intensity"])

    # 1. Load latest inference batch
    batch_dir = Path(paths_config["inference_batch_dir"])
    batch_files = sorted(batch_dir.glob("batch_*.csv"))
    if not batch_files:
        raise FileNotFoundError(f"No inference batch files found in {batch_dir}")
    latest_batch_path = batch_files[-1]
    print(f"Loading latest inference batch: {latest_batch_path}")
    curr_df = pd.read_csv(latest_batch_path)

    # 2. Build training baseline reference dataset
    print("Loading raw training baseline data for comparison...")
    ref_raw, _ = load_ieee_cis_dataset("train", config_path=config_path)
    # Exclude batch rows but keep at least 50% baseline if dataset is small
    n_inference = min(5000, len(ref_raw) // 2)
    ref_raw_baseline = ref_raw.iloc[:-n_inference].copy()

    print("Engineering features on reference baseline...")
    ref_df = engineer_features(ref_raw_baseline, config_path=config_path)

    # Resolve card column (card_type or card6)
    ref_card_col = "card_type" if "card_type" in ref_df.columns else "card6"
    curr_card_col = "card_type" if "card_type" in curr_df.columns else "card6"

    # Add hour buckets
    ref_df["hour_bucket"] = get_hour_bucket(ref_df["hour"])
    curr_df["hour_bucket"] = get_hour_bucket(curr_df["hour"])

    # 3. Define features to test
    numeric_features = ["amount", "velocity_1hr", "user_avg_amount_30d"]
    categorical_features = []
    if ref_card_col in ref_df.columns and curr_card_col in curr_df.columns:
        categorical_features.append((ref_card_col, curr_card_col))
    categorical_features.append(("hour_bucket", "hour_bucket"))

    drift_detected_flag = False
    metrics_to_log = []
    run_date_val = date.today()

    # 4. Statistical Tests
    print("Performing Kolmogorov-Smirnov tests on numerical features...")
    for feat in numeric_features:
        if feat in ref_df.columns and feat in curr_df.columns:
            ref_vals = ref_df[feat].dropna().values
            curr_vals = curr_df[feat].dropna().values
            if len(ref_vals) > 0 and len(curr_vals) > 0:
                stat, p_val = ks_2samp(ref_vals, curr_vals)
                detected = bool(p_val < drift_threshold)
                if detected:
                    drift_detected_flag = True
                print(
                    f"  {feat:20} -> KS Stat: {stat:.4f}, "
                    f"p-value: {p_val:.4e} (Drift: {detected})"
                )
                metrics_to_log.append(
                    {
                        "run_date": run_date_val,
                        "feature_name": feat,
                        "drift_score": float(stat),
                        "p_value": float(p_val),
                        "drift_detected": detected,
                        "drift_intensity": current_intensity,
                    }
                )

    print("Performing Chi-Square tests on categorical features...")
    for ref_col, curr_col in categorical_features:
        ref_vals = ref_df[ref_col].dropna().astype(str)
        curr_vals = curr_df[curr_col].dropna().astype(str)
        if len(ref_vals) > 0 and len(curr_vals) > 0:
            # Create contingency table
            ref_counts = ref_vals.value_counts()
            curr_counts = curr_vals.value_counts()
            all_cats = list(set(ref_counts.index) | set(curr_counts.index))
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_cats]
            curr_aligned = [curr_counts.get(cat, 0) for cat in all_cats]

            obs = np.array([ref_aligned, curr_aligned])
            try:
                res = chi2_contingency(obs)
                p_val = float(res.pvalue)
                stat = float(res.statistic)
                detected = bool(p_val < drift_threshold)
            except Exception:
                p_val = 1.0
                stat = 0.0
                detected = False

            if detected:
                drift_detected_flag = True
            print(
                f"  {ref_col:20} -> Chi2 Stat: {stat:.4f}, "
                f"p-value: {p_val:.4f} (Drift: {detected})"
            )
            metrics_to_log.append(
                {
                    "run_date": run_date_val,
                    "feature_name": ref_col,
                    "drift_score": stat,
                    "p_value": p_val,
                    "drift_detected": detected,
                    "drift_intensity": current_intensity,
                }
            )

    # Log results to Postgres or fallback JSON
    log_drift_metrics(metrics_to_log, database_url=database_url)

    # 5. Generate Evidently DataDriftPreset report
    print("Generating Evidently DataDriftPreset report...")
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        ref_cols_to_compare = ["amount", "velocity_1hr", "user_avg_amount_30d"]
        curr_cols_to_compare = ["amount", "velocity_1hr", "user_avg_amount_30d"]

        # Add card column separately using the resolved column name per dataset
        if ref_card_col in ref_df.columns:
            ref_cols_to_compare.append(ref_card_col)
        if curr_card_col in curr_df.columns:
            curr_cols_to_compare.append(curr_card_col)

        # Evidently requires columns to have the same names; rename curr if different
        ref_data_ev = ref_df[ref_cols_to_compare].copy()
        curr_data_ev = curr_df[curr_cols_to_compare].copy()
        if curr_card_col != ref_card_col and curr_card_col in curr_data_ev.columns:
            curr_data_ev = curr_data_ev.rename(columns={curr_card_col: ref_card_col})

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=ref_data_ev,
            current_data=curr_data_ev,
        )

        reports_dir = Path(paths_config["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_html_path = reports_dir / "drift_report.html"
        report.save_html(str(report_html_path))
        print(f"Evidently HTML report saved to {report_html_path}")
    except Exception as e:
        print(f"Warning: Failed to generate Evidently HTML report: {e}")

    report_summary = {
        "drift_detected": drift_detected_flag,
        "drift_intensity": current_intensity,
        "metrics": [
            {
                "feature": m["feature_name"],
                "drift_score": m["drift_score"],
                "p_value": m["p_value"],
                "drift_detected": m["drift_detected"],
            }
            for m in metrics_to_log
        ],
    }

    return drift_detected_flag, report_summary
