"""Data ingestion and validation utilities for IEEE-CIS data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

DatasetSplit = Literal["train", "test"]


class DataValidationError(ValueError):
    """Raised when an input dataset fails validation."""


@dataclass(frozen=True)
class IngestionSummary:
    split: str
    transaction_rows_raw: int
    identity_rows_raw: int
    rows_after_cleaning: int
    rows_after_join: int
    columns_after_join: int
    missing_transaction_ids: int
    duplicate_transaction_ids: int
    transaction_amount_nulls: int
    fraud_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise DataValidationError("Config file must contain a YAML mapping.")

    return config


def load_ieee_cis_dataset(
    split: DatasetSplit,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "config/config.yaml",
) -> tuple[pd.DataFrame, IngestionSummary]:
    """Load, validate, clean, and join one IEEE-CIS dataset split."""

    config = config or load_config(config_path)
    data_config = config["data"]
    raw_data_dir = Path(config["paths"]["raw_data_dir"])

    transaction_path, identity_path = _resolve_split_paths(
        split=split,
        raw_data_dir=raw_data_dir,
        data_config=data_config,
    )

    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path) if identity_path.exists() else pd.DataFrame()

    required_columns_key = f"required_{split}_transaction_columns"
    required_columns = data_config[required_columns_key]
    join_key = data_config["join_key"]
    target_column = data_config["target_column"]

    _validate_required_columns(transactions, required_columns, transaction_path)
    _validate_transaction_types(transactions, split, join_key, target_column)

    cleaned_transactions, cleaning_stats = _clean_transactions(
        transactions=transactions,
        join_key=join_key,
        amount_column="TransactionAmt",
    )

    if not identity.empty:
        _validate_required_columns(identity, [join_key], identity_path)
        identity = identity.drop_duplicates(subset=[join_key], keep="first")
        joined = cleaned_transactions.merge(identity, on=join_key, how="left")
    else:
        joined = cleaned_transactions.copy()

    fraud_rate = None
    if target_column in joined.columns:
        fraud_rate = float(joined[target_column].mean())

    summary = IngestionSummary(
        split=split,
        transaction_rows_raw=len(transactions),
        identity_rows_raw=len(identity),
        rows_after_cleaning=len(cleaned_transactions),
        rows_after_join=len(joined),
        columns_after_join=len(joined.columns),
        missing_transaction_ids=cleaning_stats["missing_transaction_ids"],
        duplicate_transaction_ids=cleaning_stats["duplicate_transaction_ids"],
        transaction_amount_nulls=cleaning_stats["transaction_amount_nulls"],
        fraud_rate=fraud_rate,
    )

    return joined, summary


def _resolve_split_paths(
    split: DatasetSplit,
    raw_data_dir: Path,
    data_config: dict[str, Any],
) -> tuple[Path, Path]:
    if split == "train":
        transaction_file = data_config["train_transaction_file"]
        identity_file = data_config["train_identity_file"]
    elif split == "test":
        transaction_file = data_config["test_transaction_file"]
        identity_file = data_config["test_identity_file"]
    else:
        raise DataValidationError(f"Unsupported split: {split}")

    transaction_path = raw_data_dir / transaction_file
    identity_path = raw_data_dir / identity_file

    if not transaction_path.exists():
        raise FileNotFoundError(f"Transaction file not found: {transaction_path}")

    return transaction_path, identity_path


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    source_path: Path,
) -> None:
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise DataValidationError(f"{source_path} is missing required columns: {missing}")


def _validate_transaction_types(
    transactions: pd.DataFrame,
    split: DatasetSplit,
    join_key: str,
    target_column: str,
) -> None:
    numeric_columns = ["TransactionDT", "TransactionAmt"]
    if split == "train":
        numeric_columns.append(target_column)

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(transactions[column]):
            raise DataValidationError(f"{column} must be numeric.")

    if transactions[join_key].isna().all():
        raise DataValidationError(f"{join_key} cannot be entirely null.")

    if split == "train":
        invalid_targets = sorted(set(transactions[target_column].dropna()) - {0, 1})
        if invalid_targets:
            raise DataValidationError(f"{target_column} must contain only 0 and 1.")


def _clean_transactions(
    transactions: pd.DataFrame,
    join_key: str,
    amount_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    missing_transaction_ids = int(transactions[join_key].isna().sum())
    duplicate_transaction_ids = int(transactions[join_key].duplicated().sum())
    transaction_amount_nulls = int(transactions[amount_column].isna().sum())

    cleaned = transactions.dropna(subset=[join_key]).copy()
    cleaned = cleaned.drop_duplicates(subset=[join_key], keep="first")

    if cleaned[amount_column].isna().any():
        median_amount = cleaned[amount_column].median()
        if pd.isna(median_amount):
            raise DataValidationError(f"{amount_column} cannot be entirely null.")
        cleaned[amount_column] = cleaned[amount_column].fillna(median_amount)

    return cleaned, {
        "missing_transaction_ids": missing_transaction_ids,
        "duplicate_transaction_ids": duplicate_transaction_ids,
        "transaction_amount_nulls": transaction_amount_nulls,
    }


def main() -> None:
    for split in ("train", "test"):
        _, summary = load_ieee_cis_dataset(split)
        print(summary.to_dict())


if __name__ == "__main__":
    main()
