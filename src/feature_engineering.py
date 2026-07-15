"""Feature engineering utilities for fraud detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_ingestion import load_config


def compute_rolling_nunique(
    df: pd.DataFrame,
    group_col: str,
    val_col: str,
    time_col: str,
    window_delta: float,
) -> np.ndarray:
    """Compute rolling unique counts of val_col in window_delta grouped by group_col."""
    # Ensure time is sorted to make the sliding window correct
    df_sorted = df[[group_col, val_col, time_col]].copy()
    df_sorted["idx"] = df_sorted.index

    nunique_vals = np.zeros(len(df_sorted), dtype=int)

    for _, group in df_sorted.groupby(group_col):
        times = group[time_col].values
        vals = group[val_col].astype(str).values
        idxs = group["idx"].values

        left = 0
        current_counts: dict[str, int] = {}
        for right in range(len(group)):
            # Add right element
            val_r = vals[right]
            current_counts[val_r] = current_counts.get(val_r, 0) + 1

            # Remove elements outside the window
            limit = times[right] - window_delta
            while times[left] < limit:
                val_l = vals[left]
                current_counts[val_l] -= 1
                if current_counts[val_l] == 0:
                    del current_counts[val_l]
                left += 1

            nunique_vals[idxs[right]] = len(current_counts)

    return nunique_vals


def compute_rolling_user_stats(
    df: pd.DataFrame,
    user_col: str,
    amt_col: str,
    time_col: str,
    window_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling transaction count, sum, and average amount for user_col."""
    df_sorted = df[[user_col, amt_col, time_col]].copy()
    df_sorted["idx"] = df_sorted.index

    counts = np.zeros(len(df_sorted), dtype=int)
    sums = np.zeros(len(df_sorted), dtype=float)
    means = np.zeros(len(df_sorted), dtype=float)

    for _, group in df_sorted.groupby(user_col):
        times = group[time_col].values
        amts = group[amt_col].values
        idxs = group["idx"].values

        left = 0
        current_sum = 0.0
        current_count = 0

        for right in range(len(group)):
            # Add right element
            val_r = amts[right]
            if not pd.isna(val_r):
                current_sum += float(val_r)
                current_count += 1

            # Remove elements outside the window
            limit = times[right] - window_delta
            while times[left] < limit:
                val_l = amts[left]
                if not pd.isna(val_l):
                    current_sum -= float(val_l)
                    current_count -= 1
                left += 1

            counts[idxs[right]] = current_count
            sums[idxs[right]] = current_sum
            means[idxs[right]] = (
                current_sum / current_count if current_count > 0 else 0.0
            )

    return counts, sums, means


def compute_rolling_count(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    window_delta: float,
) -> np.ndarray:
    """Compute rolling transaction count in window_delta grouped by group_col."""
    df_sorted = df[[group_col, time_col]].copy()
    df_sorted["idx"] = df_sorted.index

    counts = np.zeros(len(df_sorted), dtype=int)

    for _, group in df_sorted.groupby(group_col):
        times = group[time_col].values
        idxs = group["idx"].values

        left = 0
        current_count = 0

        for right in range(len(group)):
            current_count += 1

            # Remove elements outside the window
            limit = times[right] - window_delta
            while times[left] < limit:
                current_count -= 1
                left += 1

            counts[idxs[right]] = current_count

    return counts


def engineer_features(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "config/config.yaml",
) -> pd.DataFrame:
    """Engineer temporal, rolling, and categorical features from cleaned raw data."""
    config = config or load_config(config_path)
    feat_config = config["features"]

    # Ensure dataset is sorted by TransactionDT for point-in-time safety and reset index to prevent out-of-bounds errors on sampled data
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # 1. Temporal Features
    df["day_number"] = (df["TransactionDT"] // 86400).astype(int)
    df["hour"] = ((df["TransactionDT"] % 86400) // 3600).astype(int)
    df["day_of_week"] = (df["day_number"] % 7).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = (df["day_number"] // 30).astype(int)

    # 2. Entity Proxies
    user_cols = feat_config["user_key_columns"]
    df["user_key"] = df[user_cols].fillna("").astype(str).agg("_".join, axis=1)

    merchant_cols = feat_config["merchant_key_columns"]
    df["merchant_key"] = df[merchant_cols].fillna("").astype(str).agg("_".join, axis=1)

    # 3. Rolling Window Aggregates (30 Days = 2,592,000 seconds)
    window_30d = 30 * 86400.0

    user_counts, user_sums, user_means = compute_rolling_user_stats(
        df=df,
        user_col="user_key",
        amt_col="TransactionAmt",
        time_col="TransactionDT",
        window_delta=window_30d,
    )
    df["user_tx_count_30d"] = user_counts
    df["user_total_amount_30d"] = user_sums
    df["user_avg_amount_30d"] = user_means

    df["merchant_tx_count_30d"] = compute_rolling_count(
        df=df,
        group_col="merchant_key",
        time_col="TransactionDT",
        window_delta=window_30d,
    )

    df["merchant_unique_users_30d"] = compute_rolling_nunique(
        df=df,
        group_col="merchant_key",
        val_col="user_key",
        time_col="TransactionDT",
        window_delta=window_30d,
    )

    # 4. Velocity Window (1 Hour = 3,600 seconds)
    window_1h = 3600.0
    df["velocity_1hr"] = compute_rolling_count(
        df=df,
        group_col="user_key",
        time_col="TransactionDT",
        window_delta=window_1h,
    )

    # Rename TransactionAmt to amount to match planned feature names
    df = df.rename(columns={"TransactionAmt": "amount"})

    # 5. Select final feature set — drop raw anonymised V/C/D/M/id columns.
    # We built our own interpretable aggregates above, so V1-V339, device id_*,
    # and the raw C/D/M features are no longer needed.
    # Reduces the matrix from ~450 cols (~1.88 GB) to ~30 cols (~0.12 GB).
    target_col = config["data"]["target_column"]

    keep_cols = [
        # Identifiers (dropped later in preprocessing)
        "TransactionID",
        "TransactionDT",
        # Target (only in train split)
        target_col,
        # Transaction amount
        "amount",
        # Temporal features
        "day_number",
        "hour",
        "day_of_week",
        "is_weekend",
        "month",
        # Entity proxy keys (dropped later in preprocessing)
        "user_key",
        "merchant_key",
        # Rolling user aggregates
        "user_tx_count_30d",
        "user_total_amount_30d",
        "user_avg_amount_30d",
        # Rolling merchant aggregates
        "merchant_tx_count_30d",
        "merchant_unique_users_30d",
        # Velocity
        "velocity_1hr",
        # Card and address signals
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        # Categorical columns (OHE'd in preprocessing)
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        # Drift batch marker (only during batch generation)
        "is_inference_batch",
    ]

    # Keep only columns that actually exist in the dataframe
    final_cols = [c for c in keep_cols if c in df.columns]
    return df[final_cols]
