"""Controlled drift simulation for static IEEE-CIS data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_ingestion import load_config


def simulate_drift(
    df: pd.DataFrame,
    drift_intensity: float,
    config_path: str | Path = "config/config.yaml",
) -> pd.DataFrame:
    """Apply distribution shifts to simulate data drift based on drift intensity.

    Shifts:
    - Amount (TransactionAmt / amount): increases by up to 60%.
    - Velocity (velocity_1hr): increases by up to 200%.
    - Fraud Rate (isFraud): shifts from baseline (~3.5%) to max (~8.0%).
    - Fraud Time: shifts fraud transaction hours toward daytime (9:00 - 17:00).
    - Merchant Patterns: concentrates ProductCD, card4, P_emaildomain to trigger
      merchant transaction volume spikes.
    """
    if drift_intensity <= 0.0:
        return df.copy()

    df = df.copy()
    config = load_config(config_path)
    random_state = config["project"]["random_state"]
    rng = np.random.default_rng(random_state)

    # 1. Amount Drift
    amt_cols = ["TransactionAmt", "amount"]
    for col in amt_cols:
        if col in df.columns:
            df[col] = df[col] * (1.0 + 0.60 * drift_intensity)

    # 2. Velocity Drift (if already engineered)
    if "velocity_1hr" in df.columns:
        df["velocity_1hr"] = df["velocity_1hr"] * (1.0 + 2.0 * drift_intensity)

    # 3. Fraud Rate Drift (if labels exist)
    target_col = config["data"]["target_column"]
    if target_col in df.columns:
        curr_rate = float(df[target_col].mean())
        base_rate = 0.035
        max_rate = 0.080
        target_rate = base_rate + drift_intensity * (max_rate - base_rate)

        if curr_rate < target_rate:
            p_flip = (target_rate - curr_rate) / (1.0 - curr_rate)
            clean_mask = df[target_col] == 0
            # Flip randomly according to p_flip
            flips = rng.random(len(df)) < p_flip
            df.loc[clean_mask & flips, target_col] = 1

    # 4. Hour distribution of fraud transactions toward daytime (9:00 - 17:00)
    if "TransactionDT" in df.columns and target_col in df.columns:
        fraud_mask = df[target_col] == 1
        n_fraud = int(fraud_mask.sum())
        if n_fraud > 0:
            # Shift hour of fraud transactions with probability drift_intensity
            should_shift = rng.random(n_fraud) < drift_intensity
            fraud_indices = df[fraud_mask].index

            for i, idx in enumerate(fraud_indices):
                if should_shift[i]:
                    dt = df.loc[idx, "TransactionDT"]
                    day_num = dt // 86400
                    old_tod_seconds = dt % 86400

                    # Sample daytime hour (mean 12, std 2, clipped 9 to 17)
                    sampled_hour = rng.normal(12.0, 2.0)
                    sampled_hour = int(np.clip(sampled_hour, 9, 17))

                    new_tod_seconds = sampled_hour * 3600 + (old_tod_seconds % 3600)
                    df.loc[idx, "TransactionDT"] = day_num * 86400 + new_tod_seconds

    # 5. Merchant Volume Spikes (ProductCD, card4, P_emaildomain concentration)
    merchant_cols = ["ProductCD", "card4", "P_emaildomain"]
    for col in merchant_cols:
        if col in df.columns:
            if col == "ProductCD":
                common_val = "W"
            elif col == "card4":
                common_val = "visa"
            else:
                common_val = "gmail.com"
            mask = rng.random(len(df)) < (0.30 * drift_intensity)
            df.loc[mask, col] = common_val

    # Also shift merchant metrics if they are already engineered
    merchant_metrics = ["merchant_tx_count_30d", "merchant_unique_users_30d"]
    for metric in merchant_metrics:
        if metric in df.columns:
            df[metric] = df[metric] * (1.0 + 1.5 * drift_intensity)

    return df
