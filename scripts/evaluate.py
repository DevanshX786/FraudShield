"""Run FraudShield model evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
import joblib

# Allow running as `python scripts/evaluate.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_ingestion import load_config  # noqa: E402
from src.model_evaluation import evaluate_model  # noqa: E402


def main() -> None:
    print("=== Launching FraudShield Model Evaluation Pipeline ===")
    config = load_config("config/config.yaml")
    models_dir = Path(config["paths"]["models_dir"])
    model_path = models_dir / "xgboost_model.joblib"

    if not model_path.exists():
        print(
            f"Error: Model file not found at {model_path}. "
            "Please run training first."
        )
        sys.exit(1)

    try:
        model = joblib.load(model_path)
        report = evaluate_model(model)

        metrics = report["metrics"]
        print("=== Evaluation Pipeline Completed Successfully ===")
        print(f"Optimal Threshold: {metrics['threshold']:.2f}")
        print(f"Test F1-Score:     {metrics['f1_score']:.4f}")
        print(f"Test Precision:     {metrics['precision_score']:.4f}")
        print(f"Test Recall:        {metrics['recall_score']:.4f}")
        print(f"Test ROC-AUC:       {metrics['roc_auc_score']:.4f}")
    except Exception as e:
        print(f"Error executing evaluation pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
