"""Step 3: Run feature engineering + preprocessing on the training dataset.

Saves to:
    data/processed/  — train/val/test CSVs + feature_list.json
    models/          — one_hot_encoder.joblib, imputer.joblib, scaler.joblib
"""

from __future__ import annotations

import sys
import time

from src.data_ingestion import load_ieee_cis_dataset
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features


def main() -> None:
    t0 = time.time()
    print("=== Step 1: Loading raw training dataset ===")
    df, summary = load_ieee_cis_dataset("train")
    print(f"Rows after cleaning+join : {summary.rows_after_join:,}")
    print(f"Fraud rate               : {summary.fraud_rate:.4f}  (expected ~0.035)")
    print(f"Elapsed: {time.time() - t0:.1f}s\n")

    t1 = time.time()
    print("=== Step 2: Engineering features ===")
    df_eng = engineer_features(df)
    print(f"Engineered shape: {df_eng.shape}")
    print(f"Elapsed: {time.time() - t1:.1f}s\n")

    t2 = time.time()
    print("=== Step 3: Preprocessing (split / OHE / impute / scale / SMOTE) ===")
    splits = preprocess_data(df_eng, is_train=True)
    print(f"X_train (post-SMOTE): {splits['X_train'].shape}")
    print(f"X_val               : {splits['X_val'].shape}")
    print(f"X_test              : {splits['X_test'].shape}")
    print(f"Elapsed: {time.time() - t2:.1f}s\n")

    total = time.time() - t0
    print(f"=== Preprocessing complete in {total:.0f}s ===")
    print("Saved CSVs  -> data/processed/")
    print("Saved models -> models/ (ohe, imputer, scaler)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
