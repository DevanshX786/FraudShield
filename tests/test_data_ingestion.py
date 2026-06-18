from pathlib import Path

import pandas as pd
import pytest

from src.data_ingestion import DataValidationError, load_ieee_cis_dataset


def _base_config(raw_data_dir: Path) -> dict:
    return {
        "paths": {"raw_data_dir": str(raw_data_dir)},
        "data": {
            "train_transaction_file": "train_transaction.csv",
            "train_identity_file": "train_identity.csv",
            "test_transaction_file": "test_transaction.csv",
            "test_identity_file": "test_identity.csv",
            "target_column": "isFraud",
            "join_key": "TransactionID",
            "required_train_transaction_columns": [
                "TransactionID",
                "isFraud",
                "TransactionDT",
                "TransactionAmt",
                "ProductCD",
                "card1",
                "card2",
                "card3",
                "card4",
                "card5",
                "card6",
            ],
            "required_test_transaction_columns": [
                "TransactionID",
                "TransactionDT",
                "TransactionAmt",
                "ProductCD",
                "card1",
                "card2",
                "card3",
                "card4",
                "card5",
                "card6",
            ],
        },
    }


def _transaction_rows(include_target: bool = True) -> list[dict]:
    rows = [
        {
            "TransactionID": 1,
            "TransactionDT": 100,
            "TransactionAmt": 10.0,
            "ProductCD": "W",
            "card1": 1001,
            "card2": 200.0,
            "card3": 150.0,
            "card4": "visa",
            "card5": 226.0,
            "card6": "debit",
        },
        {
            "TransactionID": 2,
            "TransactionDT": 200,
            "TransactionAmt": None,
            "ProductCD": "C",
            "card1": 1002,
            "card2": 201.0,
            "card3": 150.0,
            "card4": "mastercard",
            "card5": 224.0,
            "card6": "credit",
        },
        {
            "TransactionID": 2,
            "TransactionDT": 250,
            "TransactionAmt": 999.0,
            "ProductCD": "C",
            "card1": 1002,
            "card2": 201.0,
            "card3": 150.0,
            "card4": "mastercard",
            "card5": 224.0,
            "card6": "credit",
        },
        {
            "TransactionID": None,
            "TransactionDT": 300,
            "TransactionAmt": 30.0,
            "ProductCD": "R",
            "card1": 1003,
            "card2": 202.0,
            "card3": 150.0,
            "card4": "visa",
            "card5": 226.0,
            "card6": "debit",
        },
    ]
    if include_target:
        targets = [0, 1, 1, 0]
        for row, target in zip(rows, targets):
            row["isFraud"] = target
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_train_dataset_cleans_and_joins_identity(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config = _base_config(raw_dir)

    _write_csv(raw_dir / "train_transaction.csv", _transaction_rows())
    _write_csv(
        raw_dir / "train_identity.csv",
        [
            {"TransactionID": 1, "DeviceType": "desktop"},
            {"TransactionID": 2, "DeviceType": "mobile"},
        ],
    )

    dataframe, summary = load_ieee_cis_dataset("train", config=config)

    assert len(dataframe) == 2
    assert list(dataframe["TransactionID"]) == [1.0, 2.0]
    assert dataframe.loc[dataframe["TransactionID"] == 2.0, "TransactionAmt"].item() == 10.0
    assert "DeviceType" in dataframe.columns
    assert summary.transaction_rows_raw == 4
    assert summary.identity_rows_raw == 2
    assert summary.missing_transaction_ids == 1
    assert summary.duplicate_transaction_ids == 1
    assert summary.transaction_amount_nulls == 1
    assert summary.fraud_rate == 0.5


def test_load_test_dataset_does_not_require_target(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config = _base_config(raw_dir)

    _write_csv(raw_dir / "test_transaction.csv", _transaction_rows(include_target=False))

    dataframe, summary = load_ieee_cis_dataset("test", config=config)

    assert "isFraud" not in dataframe.columns
    assert len(dataframe) == 2
    assert summary.fraud_rate is None


def test_missing_required_column_raises_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config = _base_config(raw_dir)
    rows = _transaction_rows()
    for row in rows:
        row.pop("card6")
    _write_csv(raw_dir / "train_transaction.csv", rows)

    with pytest.raises(DataValidationError, match="card6"):
        load_ieee_cis_dataset("train", config=config)


def test_non_numeric_amount_raises_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config = _base_config(raw_dir)
    rows = _transaction_rows()
    rows[0]["TransactionAmt"] = "not-a-number"
    _write_csv(raw_dir / "train_transaction.csv", rows)

    with pytest.raises(DataValidationError, match="TransactionAmt must be numeric"):
        load_ieee_cis_dataset("train", config=config)


def test_invalid_train_target_raises_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config = _base_config(raw_dir)
    rows = _transaction_rows()
    rows[0]["isFraud"] = 2
    _write_csv(raw_dir / "train_transaction.csv", rows)

    with pytest.raises(DataValidationError, match="isFraud must contain only 0 and 1"):
        load_ieee_cis_dataset("train", config=config)
