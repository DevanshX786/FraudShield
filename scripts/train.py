"""Run the FraudShield training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/train.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_training import train_model  # noqa: E402


def main() -> None:
    print("=== Launching FraudShield Model Training Pipeline ===")
    try:
        model, run_id = train_model()
        print("=== Training Pipeline Completed Successfully ===")
        print(f"Best Run ID: {run_id}")
    except Exception as e:
        print(f"Error executing training pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
