"""Tests for feature engineering and preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import engineer_features
from src.data_preprocessing import preprocess_data


def _mock_config(raw_data_dir: Path) -> dict:
    return {
        "project": {
            "name": "test-project",
            "random_state": 42,
        },
        "paths": {
            "raw_data_dir": str(raw_data_dir),
            "processed_data_dir": str(raw_data_dir / "processed"),
            "models_dir": str(raw_data_dir / "models"),
        },
        "data": {
            "target_column": "isFraud",
            "join_key": "TransactionID",
        },
        "features": {
            "user_key_columns": ["card1", "card2"],
            "merchant_key_columns": ["ProductCD", "card4"],
            "categorical_columns": ["ProductCD", "card4", "card6"],
        },
        "split": {
            "train_size": 0.60,
            "validation_size": 0.20,
            "test_size": 0.20,
            # Use full 50/50 SMOTE in unit tests (small dataset, no OOM risk)
            "use_smote": True,
            "smote_sampling_strategy": 1.0,
        },
    }


def test_engineer_features_temporal_and_keys(tmp_path: Path) -> None:
    config = _mock_config(tmp_path)
    # Day 0: 0 seconds (hour=0, day_of_week=0, is_weekend=0)
    # Day 5 (Saturday): 5 * 86400 = 432000 seconds (hour=0, day_of_week=5, is_weekend=1)
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [0, 432000],
            "TransactionAmt": [100.0, 200.0],
            "card1": [1000, 2000],
            "card2": [100.0, 200.0],
            "ProductCD": ["W", "C"],
            "card4": ["visa", "mastercard"],
            "card6": ["debit", "credit"],
        }
    )

    df_feats = engineer_features(df, config=config)

    # Check temporal features
    assert list(df_feats["hour"]) == [0, 0]
    assert list(df_feats["day_of_week"]) == [0, 5]
    assert list(df_feats["is_weekend"]) == [0, 1]

    # Check proxy keys
    assert list(df_feats["user_key"]) == ["1000_100.0", "2000_200.0"]
    assert list(df_feats["merchant_key"]) == ["W_visa", "C_mastercard"]


def test_engineer_features_rolling_windows(tmp_path: Path) -> None:
    config = _mock_config(tmp_path)

    # We want to test user rolling aggregates.
    # User A transactions:
    # 1. T = 0, amt = 10
    # 2. T = 100, amt = 20
    # 3. T = 2,592,100 (30 days + 100 seconds). The transaction at T=0 is
    #    outside 30D window, T=100 is inside.
    # Total window duration = 30 * 86400 = 2,592,000 seconds.
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [0, 100, 2592100],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "card1": [1000, 1000, 1000],
            "card2": [100.0, 100.0, 100.0],
            "ProductCD": ["W", "W", "W"],
            "card4": ["visa", "visa", "visa"],
            "card6": ["debit", "debit", "debit"],
        }
    )

    df_feats = engineer_features(df, config=config)

    # User tx count 30d
    # Row 1 (T=0): 1 tx (T=0)
    # Row 2 (T=100): 2 txs (T=0, T=100)
    # Row 3 (T=2592100): 2 txs (T=100, T=2592100) -> T=0 is 2,592,100s ago,
    #    which is > 2,592,000s
    assert list(df_feats["user_tx_count_30d"]) == [1, 2, 2]

    # User total amount 30d
    # Row 1: 10
    # Row 2: 30
    # Row 3: 50 (20 + 30)
    assert list(df_feats["user_total_amount_30d"]) == [10.0, 30.0, 50.0]

    # User avg amount 30d
    # Row 1: 10 / 1 = 10
    # Row 2: 30 / 2 = 15
    # Row 3: 50 / 2 = 25
    assert list(df_feats["user_avg_amount_30d"]) == [10.0, 15.0, 25.0]

    # Velocity 1 hour (3600 seconds)
    # Row 1 (T=0): 1 tx (T=0)
    # Row 2 (T=100): 2 txs (T=0, T=100) -> 100s difference
    # Row 3 (T=2592100): 1 tx (T=2592100) -> T=100 is way outside 1 hour
    assert list(df_feats["velocity_1hr"]) == [1, 2, 1]


def test_preprocess_data_splits_and_smote(tmp_path: Path) -> None:
    config = _mock_config(tmp_path)

    # Build a mock dataset with enough rows to split 60/20/20.
    # Total rows = 20. Train = 12, Val = 4, Test = 4.
    # Target counts: Clean: 14 rows, Fraud: 6 rows.
    # To run SMOTE, the minority class in the training split (60% of 6 = 3.6 -> 4 rows)
    # Wait, SMOTE's default k_neighbors is 5, which requires >= 6 minority samples.
    # Let's create a dataset of 30 rows with 21 clean and 9 fraud.
    # Train = 18 (13 clean, 5 fraud). If train has 5 fraud, we need k_neighbors <= 4.
    # Or create a larger dataset: 40 rows (28 clean, 12 fraud).
    # Train (60% of 40 = 24 rows) -> 17 clean, 7 fraud.
    # 7 fraud samples in train is > 5 k_neighbors, so SMOTE runs perfectly!
    np.random.seed(42)
    n_rows = 40
    is_fraud = [0] * 28 + [1] * 12

    df = pd.DataFrame(
        {
            "TransactionID": list(range(n_rows)),
            "TransactionDT": sorted(np.random.randint(0, 100000, n_rows)),
            "TransactionAmt": np.random.uniform(5.0, 1000.0, n_rows),
            "card1": np.random.randint(1000, 2000, n_rows),
            "card2": np.random.uniform(100.0, 500.0, n_rows),
            "ProductCD": np.random.choice(["W", "C", "R"], n_rows),
            "card4": np.random.choice(["visa", "mastercard"], n_rows),
            "card6": np.random.choice(["debit", "credit"], n_rows),
            "isFraud": is_fraud,
        }
    )

    # Engineer features first
    df_feats = engineer_features(df, config=config)

    # Run preprocessing
    splits = preprocess_data(df_feats, is_train=True, config=config)

    # Check outputs exist and SMOTE has oversampled the training set
    assert isinstance(splits, dict)
    assert "X_train" in splits
    assert "y_train" in splits
    assert "X_val" in splits
    assert "y_val" in splits
    assert "X_test" in splits
    assert "y_test" in splits

    # SMOTE balances the classes in training set to 50/50.
    # The clean training set was 17 rows, so after SMOTE, both classes should
    # have 17 rows, totaling 34.
    y_train = splits["y_train"]
    assert len(y_train) == 34
    assert (y_train == 0).sum() == 17
    assert (y_train == 1).sum() == 17

    # Validation and test sets should remain unbalanced (40 * 0.20 = 8 rows each)
    assert len(splits["X_val"]) == 8
    assert len(splits["X_test"]) == 8
    assert (splits["y_val"] == 1).sum() < len(splits["y_val"])

    # Verify files saved on disk
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    assert (processed_dir / "train_feats.csv").exists()
    assert (processed_dir / "feature_list.json").exists()
    assert (models_dir / "scaler.joblib").exists()
    assert (models_dir / "imputer.joblib").exists()
    assert (models_dir / "one_hot_encoder.joblib").exists()

    # Verify inference mode preprocess
    # Create a small inference df
    df_inf = df.head(5).copy()
    df_inf_feats = engineer_features(df_inf, config=config)
    X_inf = preprocess_data(df_inf_feats, is_train=False, config=config)

    assert isinstance(X_inf, pd.DataFrame)
    assert len(X_inf) == 5
    assert list(X_inf.columns) == list(splits["X_train"].columns)
