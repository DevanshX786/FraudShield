"""Run FraudShield drift detection and conditional retraining."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/check_drift.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drift_detector import detect_drift  # noqa: E402
from src.retraining_pipeline import retrain_pipeline  # noqa: E402


def main() -> None:
    """Run drift detection and trigger retraining if drift is detected."""
    print("Starting drift check...")
    try:
        drift_detected, summary = detect_drift()
    except Exception as e:
        print(f"Error executing drift detection: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Drift Summary ---")
    print(f"Drift Detected: {drift_detected}")
    print(f"Drift Intensity: {summary['drift_intensity']}")
    print("Feature Metrics:")
    for metric in summary["metrics"]:
        print(
            f"  - {metric['feature']}: Drift Score = {metric['drift_score']:.4f}, "
            f"p-value = {metric['p_value']:.4e} (Drift = {metric['drift_detected']})"
        )

    if drift_detected:
        print(
            "\nALERT: Data drift detected! "
            "Triggering automated retraining pipeline..."
        )
        try:
            retrain_pipeline()
            print("Automated retraining pipeline execution complete.")
        except Exception as e:
            print(f"Error during automated retraining: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\nNo significant data drift detected. Model retraining not required.")


if __name__ == "__main__":
    main()
