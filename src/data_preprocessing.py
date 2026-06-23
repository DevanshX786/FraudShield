"""Train, validation, and test preprocessing utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_ingestion import load_config


def preprocess_data(
    df: pd.DataFrame,
    is_train: bool = True,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "config/config.yaml",
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """Preprocess dataset: splitting, imputation, scaling, and SMOTE (train only)."""
    config = config or load_config(config_path)
    data_config = config["data"]
    split_config = config["split"]
    paths_config = config["paths"]

    target_col = data_config["target_column"]
    cat_cols = config["features"]["categorical_columns"]

    # Ensure processed data and model directories exist
    processed_dir = Path(paths_config["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(paths_config["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fill NaNs in categorical columns and convert to string
    df = df.copy()
    existing_cat_cols = [col for col in cat_cols if col in df.columns]
    for col in existing_cat_cols:
        df[col] = df[col].fillna("unknown").astype(str)

    # 2. Separate target if present
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col] if is_train else None
    elif is_train:
        raise ValueError(
            f"Target column '{target_col}' is missing from the dataframe. "
            "Cannot preprocess in train mode without labels. "
            "Ensure the input dataframe contains the isFraud column before calling "
            "preprocess_data(is_train=True)."
        )
    else:
        X = df
        y = None

    # 3. Identify and drop non-feature identifier columns
    cols_to_drop = [
        "TransactionID",
        "TransactionDT",
        "user_key",
        "merchant_key",
        "day_number",
    ]
    existing_drops = [col for col in cols_to_drop if col in X.columns]
    X_feats = X.drop(columns=existing_drops)

    # Separate numerical and categorical columns
    num_cols = [col for col in X_feats.columns if col not in existing_cat_cols]

    if is_train and y is not None:
        # 4. Perform Stratified Train/Val/Test Split (70/15/15)
        # First split off train (70%) and temp (30%)
        test_size_temp = split_config["validation_size"] + split_config["test_size"]
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_feats,
            y,
            test_size=test_size_temp,
            stratify=y,
            random_state=config["project"]["random_state"],
        )

        # Split temp into val (50% of temp = 15%) and test (50% of temp = 15%)
        val_ratio_of_temp = split_config["validation_size"] / test_size_temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1.0 - val_ratio_of_temp,
            stratify=y_temp,
            random_state=config["project"]["random_state"],
        )

        # 5. One-Hot Encode Categorical Columns
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
        ohe.fit(X_train[existing_cat_cols])
        joblib.dump(ohe, models_dir / "one_hot_encoder.joblib")

        def transform_ohe(df_split: pd.DataFrame) -> pd.DataFrame:
            encoded_arr = ohe.transform(df_split[existing_cat_cols])
            encoded_cols = ohe.get_feature_names_out(existing_cat_cols)
            encoded_df = pd.DataFrame(
                encoded_arr, columns=encoded_cols, index=df_split.index
            )
            # Drop original categorical columns and concat encoded ones
            return pd.concat(
                [df_split.drop(columns=existing_cat_cols), encoded_df], axis=1
            )

        X_train_enc = transform_ohe(X_train)
        X_val_enc = transform_ohe(X_val)
        X_test_enc = transform_ohe(X_test)

        # 6. Impute Missing Values (using median computed on train split only)
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_train_enc)
        joblib.dump(imputer, models_dir / "imputer.joblib")

        X_train_imp = pd.DataFrame(
            imputer.transform(X_train_enc),
            columns=X_train_enc.columns,
            index=X_train_enc.index,
        )
        X_val_imp = pd.DataFrame(
            imputer.transform(X_val_enc),
            columns=X_val_enc.columns,
            index=X_val_enc.index,
        )
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test_enc),
            columns=X_test_enc.columns,
            index=X_test_enc.index,
        )

        # 7. Scale Numerical Features
        scaler = StandardScaler()
        # Fit only on the numerical features of the training set.
        # We find the intersection of numerical column names with what exists
        # in the encoded df.
        existing_num_cols = [col for col in num_cols if col in X_train_imp.columns]
        scaler.fit(X_train_imp[existing_num_cols])
        joblib.dump(scaler, models_dir / "scaler.joblib")

        def transform_scaler(df_imp: pd.DataFrame) -> pd.DataFrame:
            df_scaled = df_imp.copy()
            df_scaled[existing_num_cols] = scaler.transform(df_imp[existing_num_cols])
            return df_scaled

        X_train_scaled = transform_scaler(X_train_imp)
        X_val_scaled = transform_scaler(X_val_imp)
        X_test_scaled = transform_scaler(X_test_imp)

        # 8. Apply oversampling to training set only.
        # sampling_strategy controls how far to rebalance:
        #   0.3 = minority grows to 30% of majority (uses ~0.5 GB)
        #   1.0 = full 50/50 balance (requires ~1.75 GB on 590k rows)
        # Falls back to RandomOverSampler on MemoryError (very low RAM machines).
        random_state = config["project"]["random_state"]
        use_smote = split_config.get("use_smote", True)
        smote_strategy = float(split_config.get("smote_sampling_strategy", 0.3))

        if use_smote:
            try:
                print(
                    f"Applying SMOTE (sampling_strategy={smote_strategy}) "
                    "to training set..."
                )
                sampler = SMOTE(
                    sampling_strategy=smote_strategy,
                    random_state=random_state,
                )
                X_train_res, y_train_res = sampler.fit_resample(X_train_scaled, y_train)
                print(
                    f"SMOTE complete: {X_train_res.shape[0]:,} training rows "
                    f"(was {X_train_scaled.shape[0]:,})."
                )
            except MemoryError:
                print(
                    "WARNING: SMOTE ran out of memory. "
                    "Falling back to RandomOverSampler (duplicates minority samples)."
                )
                sampler = RandomOverSampler(
                    sampling_strategy=smote_strategy,
                    random_state=random_state,
                )
                X_train_res, y_train_res = sampler.fit_resample(X_train_scaled, y_train)
        else:
            print("SMOTE disabled in config. Using raw training split.")
            X_train_res = X_train_scaled
            y_train_res = y_train

        # 9. Save Feature Columns List
        feature_list = list(X_train_scaled.columns)
        with open(processed_dir / "feature_list.json", "w", encoding="utf-8") as f:
            json.dump(feature_list, f, indent=4)

        # 10. Save all processed splits to disk
        X_train_res.to_csv(processed_dir / "train_feats.csv", index=False)
        y_train_res.to_csv(
            processed_dir / "train_target.csv", index=False, header=[target_col]
        )
        X_val_scaled.to_csv(processed_dir / "val_feats.csv", index=False)
        y_val.to_csv(processed_dir / "val_target.csv", index=False, header=[target_col])
        X_test_scaled.to_csv(processed_dir / "test_feats.csv", index=False)
        y_test.to_csv(
            processed_dir / "test_target.csv", index=False, header=[target_col]
        )

        return {
            "X_train": X_train_res,
            "y_train": y_train_res,
            "X_val": X_val_scaled,
            "y_val": y_val,
            "X_test": X_test_scaled,
            "y_test": y_test,
        }

    else:
        # Inference mode: load fitted transformers
        ohe = joblib.load(models_dir / "one_hot_encoder.joblib")
        imputer = joblib.load(models_dir / "imputer.joblib")
        scaler = joblib.load(models_dir / "scaler.joblib")

        # Load target feature columns list
        with open(processed_dir / "feature_list.json", "r", encoding="utf-8") as f:
            feature_list = json.load(f)

        # One-hot encode
        encoded_arr = ohe.transform(X_feats[existing_cat_cols])
        encoded_cols = ohe.get_feature_names_out(existing_cat_cols)
        encoded_df = pd.DataFrame(
            encoded_arr, columns=encoded_cols, index=X_feats.index
        )
        X_enc = pd.concat([X_feats.drop(columns=existing_cat_cols), encoded_df], axis=1)

        # Impute
        X_imp = pd.DataFrame(
            imputer.transform(X_enc), columns=X_enc.columns, index=X_enc.index
        )

        # Scale
        existing_num_cols = [col for col in num_cols if col in X_imp.columns]
        X_scaled = X_imp.copy()
        X_scaled[existing_num_cols] = scaler.transform(X_imp[existing_num_cols])

        # Align with target feature list (handling missing or extra dummy columns)
        X_final = X_scaled.reindex(columns=feature_list, fill_value=0.0)

        return X_final
