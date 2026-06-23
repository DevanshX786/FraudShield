"""Tests for controlled drift simulation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.drift_simulator import simulate_drift


def _write_mock_config(path: Path) -> None:
    config_dict = {
        "project": {
            "name": "test-drift",
            "random_state": 42,
        },
        "data": {
            "target_column": "isFraud",
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)


def test_simulate_drift_no_intensity(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    _write_mock_config(config_file)

    df = pd.DataFrame(
        {
            "TransactionAmt": [10.0, 20.0, 30.0],
            "isFraud": [0, 0, 0],
        }
    )

    drifted = simulate_drift(df, drift_intensity=0.0, config_path=config_file)
    pd.testing.assert_frame_equal(df, drifted)


def test_simulate_drift_with_intensity(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    _write_mock_config(config_file)

    # Balanced dataset
    df = pd.DataFrame(
        {
            "TransactionAmt": [100.0] * 200,
            "isFraud": [0] * 200,
            "TransactionDT": [1000 + i * 1000 for i in range(200)],
            "ProductCD": ["H"] * 200,
            "card4": ["discover"] * 200,
            "P_emaildomain": ["yahoo.com"] * 200,
        }
    )

    # Apply 1.0 intensity drift
    drifted = simulate_drift(df, drift_intensity=1.0, config_path=config_file)

    # 1. Amount should increase by 60%
    assert drifted["TransactionAmt"].iloc[0] == 160.0

    # 2. Some 0s should be flipped to 1s
    assert drifted["isFraud"].sum() > 0

    # 3. ProductCD/card4/P_emaildomain should have some common values
    assert (
        "W" in drifted["ProductCD"].values
        or "visa" in drifted["card4"].values
        or "gmail.com" in drifted["P_emaildomain"].values
    )
