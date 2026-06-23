"""Generate a simulated inference batch with controlled drift."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Allow running as `python scripts/generate_inference_batch.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_ingestion import load_config, load_ieee_cis_dataset  # noqa: E402
from src.drift_simulator import simulate_drift  # noqa: E402
from src.feature_engineering import engineer_features  # noqa: E402


def main() -> None:
    """Generate and save a feature-engineered drifted inference batch."""
    print("Loading config...")
    config = load_config()
    paths_config = config["paths"]
    drift_config = config["drift_simulation"]

    intensity = float(drift_config["current_intensity"])
    print(f"Current drift intensity: {intensity}")

    # Create inference batches directory if it doesn't exist
    batch_dir = Path(paths_config["inference_batch_dir"])
    batch_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw training dataset split as base...")
    df_raw, _ = load_ieee_cis_dataset("train")

    # Sort by TransactionDT to ensure temporal correctness
    df_raw = df_raw.sort_values("TransactionDT").reset_index(drop=True)

    # We take the last 5000 rows to simulate the most recent transaction batch
    n_inference = min(5000, len(df_raw))
    print(f"Selecting the last {n_inference} rows for the inference batch...")

    # Set markers
    df_raw["is_inference_batch"] = False
    df_raw.iloc[-n_inference:, df_raw.columns.get_loc("is_inference_batch")] = True

    # Separate baseline and candidate batch
    baseline_df = df_raw[~df_raw["is_inference_batch"]].copy()
    candidate_df = df_raw[df_raw["is_inference_batch"]].copy()

    # Apply drift only to the candidate inference batch raw columns
    print("Simulating drift on the candidate batch...")
    drifted_candidate_df = simulate_drift(candidate_df, drift_intensity=intensity)

    # Combine back to correctly compute rolling aggregates (O(N) rolling window)
    combined_df = concat_and_align([baseline_df, drifted_candidate_df])

    print("Engineering features on the combined dataset...")
    engineered_df = engineer_features(combined_df)

    # Extract the drifted engineered inference batch rows
    inference_batch = engineered_df[engineered_df["is_inference_batch"]].copy()
    inference_batch = inference_batch.drop(columns=["is_inference_batch"])

    # Format filename: batch_YYYYMMDD.csv
    date_str = datetime.now().strftime("%Y%m%d")
    out_file = batch_dir / f"batch_{date_str}.csv"

    # Save to disk
    inference_batch.to_csv(out_file, index=False)
    print(
        f"Successfully saved drifted inference batch to {out_file} "
        f"({len(inference_batch)} rows)."
    )


def concat_and_align(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate DataFrames while preserving columns and sorting by TransactionDT."""
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    return combined.sort_values("TransactionDT").reset_index(drop=True)


if __name__ == "__main__":
    main()
