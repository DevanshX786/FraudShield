"""Model evaluation and threshold tuning helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_ingestion import load_config


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types in json.dump."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def evaluate_model(
    model: Any,
    config_path: str | Path = "config/config.yaml",
) -> dict[str, Any]:
    """Tune threshold, compute test split metrics, compute SHAP, and save report."""
    config = load_config(config_path)
    paths_config = config["paths"]
    model_config = config["model"]

    processed_dir = Path(paths_config["processed_data_dir"])
    reports_dir = Path(paths_config["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Processed splits
    X_val = pd.read_csv(processed_dir / "val_feats.csv")
    y_val = pd.read_csv(processed_dir / "val_target.csv").iloc[:, 0]
    X_test = pd.read_csv(processed_dir / "test_feats.csv")
    y_test = pd.read_csv(processed_dir / "test_target.csv").iloc[:, 0]

    # 2. Get predictions and probabilities on validation split
    probs_val = model.predict_proba(X_val)[:, 1]

    # 3. Threshold Tuning (0.3 to 0.7 in steps of 0.05)
    best_threshold = 0.5
    best_f1 = -1.0

    thresholds = np.arange(
        model_config["threshold_search"]["min"],
        model_config["threshold_search"]["max"] + 0.01,  # include max
        model_config["threshold_search"]["step"],
    )

    print("Tuning classification threshold on validation split...")
    for thresh in thresholds:
        preds = (probs_val >= thresh).astype(int)
        f1 = float(f1_score(y_val, preds))
        print(f"  Threshold: {thresh:.2f} -> Validation F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thresh)

    print(
        f"Optimal Threshold Selected: {best_threshold:.2f} "
        f"with Val F1: {best_f1:.4f}"
    )

    # 4. Evaluate on Test Split using Optimal Threshold
    probs_test = model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= best_threshold).astype(int)

    test_f1 = float(f1_score(y_test, preds_test))
    test_precision = float(precision_score(y_test, preds_test))
    test_recall = float(recall_score(y_test, preds_test))
    test_roc_auc = float(roc_auc_score(y_test, probs_test))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, preds_test).ravel()

    metrics = {
        "f1_score": test_f1,
        "precision_score": test_precision,
        "recall_score": test_recall,
        "roc_auc_score": test_roc_auc,
        "threshold": best_threshold,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    # 5. SHAP Explainability
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    # Get SHAP values for test features
    shap_values_obj = explainer(X_test)

    # SHAP returns an Explanation object in newer versions, extract values
    if hasattr(shap_values_obj, "values"):
        shap_values = shap_values_obj.values
    else:
        shap_values = shap_values_obj

    # Global SHAP Feature Importance (top 10)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_series = pd.Series(mean_abs_shap, index=X_test.columns)
    top_10_features = importance_series.sort_values(ascending=False).head(10).to_dict()

    # Local SHAP explanations for the first 5 test samples
    local_explanations = []
    for idx in range(min(5, len(X_test))):
        row_feats = X_test.iloc[idx].to_dict()
        row_shap = dict(zip(X_test.columns, shap_values[idx]))
        local_explanations.append(
            {
                "index": idx,
                "actual_label": int(y_test.iloc[idx]),
                "fraud_probability": float(probs_test[idx]),
                "prediction": "FRAUD" if probs_test[idx] >= best_threshold else "CLEAN",
                "features": row_feats,
                "shap_values": row_shap,
            }
        )

    # 6. Construct and Save Evaluation Report
    report = {
        "metrics": metrics,
        "global_shap_importance": top_10_features,
        "local_explanations_sample": local_explanations,
    }

    report_file = reports_dir / "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, cls=NumpyEncoder)
    print(f"Saved model evaluation report to {report_file}")

    return report
