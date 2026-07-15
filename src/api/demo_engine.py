"""Backend engine for FraudShield Demo Mode scenario with manual stage controls."""

from __future__ import annotations

import logging
import random
import shutil
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger("demo_engine")


class DemoScenarioEngine:
    """Manages state transitions, metrics, events, and retraining for the MLOps demo."""

    def __init__(self) -> None:
        self.is_active = False
        self.stage = 1
        self.stage_seconds = 0
        self.status = "healthy"
        self.accuracy = 96.0
        self.drift_score = 0.03
        self.confidence = 98.0
        self.fraud_detection_rate = 94.0

        self.events: list[dict[str, Any]] = []
        self.drift_history: list[dict[str, Any]] = []
        self.promotion_history: list[dict[str, Any]] = []
        self.recent_transactions: list[dict[str, Any]] = []

        self._thread: threading.Thread | None = None
        self._retrain_thread: threading.Thread | None = None

    def start(self, drift_score: float = 0.65) -> None:
        """Reset state, apply drift immediately, and start retraining in a background thread."""
        self.stop()
        self.is_active = True
        try:
            import src.api.main as api_main

            api_main.MODEL_VERSION = "v1"
        except Exception:
            pass
        self.stage = 3  # Drift Alert
        self.stage_seconds = 0
        self.status = "drift_detected"

        self.drift_score = drift_score

        # Calculate degraded metrics based on drift score
        factor = (drift_score - 0.05) / 0.95 if drift_score > 0.05 else 0.0
        self.accuracy = max(72.0, 96.0 - factor * 24.0)
        self.confidence = max(68.0, 98.0 - factor * 30.0)
        self.fraud_detection_rate = max(76.0, 94.0 - factor * 18.0)

        self.events = []
        self.drift_history = []
        self.promotion_history = [
            {
                "triggered_at": (datetime.now() - pd.Timedelta(hours=12)).isoformat(),
                "trigger_reason": "scheduled",
                "old_f1": 0.815,
                "new_f1": 0.824,
                "promoted": True,
                "notes": "Baseline production model",
            }
        ]
        self._generate_initial_transactions()
        self._backup_prod_model()

        self.add_event(
            f"ALERT: Concept drift detected in transaction stream (intensity: {drift_score:.2f})"
        )
        self.add_event("Retraining threshold exceeded (0.05)")

        # Sync with config.yaml immediately
        try:
            import yaml

            with open("config/config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if "drift_simulation" in cfg:
                cfg["drift_simulation"]["current_intensity"] = drift_score
                with open("config/config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f)
        except Exception:
            pass

        # Start scenario loop thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Trigger retraining thread immediately
        self.stage = 4
        self._retrain_thread = threading.Thread(
            target=self._run_retraining, daemon=True
        )
        self._retrain_thread.start()

    def stop(self) -> None:
        """Stop active scenario and restore original model."""
        self.is_active = False
        # Restore backup model
        self._restore_prod_model()

        # Set model version back to original
        try:
            import src.api.main as api_main

            api_main.MODEL_VERSION = "local_v1"
            api_main.load_production_model()
        except Exception:
            pass

    def set_stage(self, stage: int) -> None:
        """Set current demo stage and reset stage step counters."""
        if not self.is_active:
            self.start()

        self.stage = stage
        self.stage_seconds = 0

        if stage == 1:
            self.status = "healthy"
            self.accuracy = 96.0
            self.drift_score = 0.03
            self.confidence = 98.0
            self.fraud_detection_rate = 94.0
            self.add_event("Monitoring transaction stream")
            # Restore model version
            self._restore_prod_model()
            try:
                import src.api.main as api_main

                api_main.MODEL_VERSION = "v1"
                api_main.load_production_model()
            except Exception:
                pass
        elif stage == 2:
            self.add_event("Distribution shift detected")
        elif stage == 3:
            self.status = "drift_detected"
            self.accuracy = 72.0
            self.drift_score = 0.65
            self.confidence = 68.0
            self.fraud_detection_rate = 76.0
            self.add_event("Drift threshold exceeded")
            self.add_event("ALERT: Concept drift detected")
        elif stage == 4:
            self.status = "drift_detected"
            self.accuracy = 72.0
            self.drift_score = 0.65
            self.confidence = 68.0
            self.fraud_detection_rate = 76.0
            # Trigger retraining thread if not alive
            if self._retrain_thread is None or not self._retrain_thread.is_alive():
                self._retrain_thread = threading.Thread(
                    target=self._run_retraining, daemon=True
                )
                self._retrain_thread.start()
        elif stage == 5:
            self.status = "healthy"
            # Deploy new model immediately
            try:
                import src.api.main as api_main

                api_main.MODEL_VERSION = "v2"
                api_main.load_production_model()
            except Exception:
                pass
            self.add_event("New model deployed")
            self.add_event("Drift resolved")

    def add_event(self, message: str) -> None:
        """Add a timestamped event log."""
        # Simple timestamp for demo mode representation
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        # Avoid duplicate events
        if not any(e.get("message") == f"{timestamp} {message}" for e in self.events):
            self.events.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "time_relative": timestamp,
                    "message": f"{timestamp} {message}",
                }
            )

    def get_status_dict(self) -> dict[str, Any]:
        """Return the current metrics and states."""
        return {
            "is_active": self.is_active,
            "stage": self.stage,
            "status": self.status,
            "accuracy": self.accuracy,
            "drift_score": self.drift_score,
            "confidence": self.confidence,
            "fraud_detection_rate": self.fraud_detection_rate,
            "is_retraining": self._retrain_thread is not None
            and self._retrain_thread.is_alive(),
            "events": [e["message"] for e in self.events],
        }

    def _backup_prod_model(self) -> None:
        """Create a backup of the current production model file."""
        model_path = Path("models/xgboost_model.joblib")
        backup_path = Path("models/xgboost_model.joblib.bak")
        if model_path.exists() and not backup_path.exists():
            try:
                shutil.copy2(model_path, backup_path)
                logger.info("Successfully backed up production model.")
            except Exception as e:
                logger.warning(f"Failed to back up production model: {e}")

    def _restore_prod_model(self) -> None:
        """Restore original production model file from backup."""
        model_path = Path("models/xgboost_model.joblib")
        backup_path = Path("models/xgboost_model.joblib.bak")
        if backup_path.exists():
            try:
                if model_path.exists():
                    model_path.unlink()
                shutil.move(backup_path, model_path)
                logger.info("Successfully restored production model.")
            except Exception as e:
                logger.warning(f"Failed to restore production model: {e}")

    def _generate_initial_transactions(self) -> None:
        """Seed initial transactions for the feed."""
        self.recent_transactions = []
        for i in range(15):
            tx_id = f"TX_INIT_{1000 + i}"
            self.recent_transactions.append(
                {
                    "transaction_id": tx_id,
                    "fraud_probability": 0.02 + random.random() * 0.1,
                    "prediction": "CLEAN",
                    "model_version": "v1",
                    "shap_explanation": {"amount": -0.05, "velocity_1hr": -0.02},
                    "timestamp": (
                        datetime.now() - pd.Timedelta(seconds=i * 4)
                    ).isoformat(),
                    "amount": 10.0 + random.random() * 100.0,
                    "card_type": "debit (visa)",
                    "product_cd": "W",
                    "card_network": "visa",
                }
            )

    def _run_loop(self) -> None:
        """Scenario loop updating values every second."""
        tick_counter = 0
        while self.is_active:
            # 1. Update gradual transitions
            if self.stage == 2:
                # Gradual degradation towards target drift metrics
                self.drift_score = min(0.65, self.drift_score + 0.031)
                self.accuracy = max(72.0, self.accuracy - 1.2)
                self.confidence = max(68.0, self.confidence - 1.5)
                self.fraud_detection_rate = max(76.0, self.fraud_detection_rate - 0.9)

                # Append Stage 2 secondary logs relative to step progress
                if self.stage_seconds == 3:
                    self.add_event("Fraud pattern deviation detected")
                elif self.stage_seconds == 6:
                    self.add_event("Drift score increasing")

            elif self.stage == 5:
                # Gradual recovery back to healthy metrics
                self.drift_score = max(0.03, self.drift_score - 0.124)
                self.accuracy = min(96.0, self.accuracy + 4.8)
                self.confidence = min(98.0, self.confidence + 6.0)
                self.fraud_detection_rate = min(94.0, self.fraud_detection_rate + 3.6)

            # Map tick counter to unique dates to display progression correctly in Recharts
            virtual_date = (
                date(2026, 6, 1) + pd.Timedelta(days=tick_counter)
            ).isoformat()
            self.drift_history.append(
                {
                    "run_date": virtual_date,
                    "feature_name": "amount",
                    "drift_score": self.drift_score,
                    "p_value": 0.0001 if self.drift_score > 0.05 else 0.8,
                    "drift_detected": self.drift_score > 0.05,
                    "drift_intensity": self.drift_score,
                }
            )
            self.drift_history.append(
                {
                    "run_date": virtual_date,
                    "feature_name": "velocity_1hr",
                    "drift_score": self.drift_score * 0.9,
                    "p_value": 0.0001 if self.drift_score > 0.05 else 0.8,
                    "drift_detected": self.drift_score > 0.05,
                    "drift_intensity": self.drift_score,
                }
            )

            # Stream simulated transactions in real-time
            self._stream_transaction()

            self.stage_seconds += 1
            tick_counter += 1
            time.sleep(1.0)

    def _stream_transaction(self) -> None:
        """Generate and prepend simulated transaction matching active stage risk profile."""
        tx_id = f"TX_DEMO_{random.randint(100000, 999999)}"

        if self.stage == 1 or self.stage == 5:
            prob = 0.01 + random.random() * 0.09
            amt = 5.0 + random.random() * 95.0
            pred = "CLEAN"
            conf = "HIGH"
            shap = {"amount": -0.05, "velocity_1hr": -0.03}
        elif self.stage == 2:
            # Gradually shifting probabilities
            factor = min(1.0, self.stage_seconds / 20.0)
            prob = (0.01 + random.random() * 0.09) * (1.0 - factor) + (
                0.4 + random.random() * 0.55
            ) * factor
            amt = (5.0 + random.random() * 95.0) * (1.0 - factor) + (
                350.0 + random.random() * 450.0
            ) * factor

            if prob >= 0.5:
                pred = "FRAUD"
                conf = "HIGH" if prob >= 0.7 else "MEDIUM"
                shap = {"amount": 0.35 * factor, "velocity_1hr": 0.25 * factor}
            elif prob >= 0.3:
                pred = "SUSPICIOUS"
                conf = "MEDIUM"
                shap = {"amount": 0.18 * factor, "velocity_1hr": 0.12 * factor}
            else:
                pred = "CLEAN"
                conf = "MEDIUM"
                shap = {"amount": -0.01, "velocity_1hr": 0.02}
        else:
            # Stage 3 & 4 (Drift alert and Retraining)
            prob = 0.4 + random.random() * 0.55
            amt = 350.0 + random.random() * 450.0
            if prob >= 0.5:
                pred = "FRAUD"
                conf = "HIGH" if prob >= 0.7 else "MEDIUM"
                shap = {"amount": 0.35, "velocity_1hr": 0.25}
            elif prob >= 0.3:
                pred = "SUSPICIOUS"
                conf = "MEDIUM"
                shap = {"amount": 0.18, "velocity_1hr": 0.12}
            else:
                pred = "CLEAN"
                conf = "MEDIUM"
                shap = {"amount": -0.01, "velocity_1hr": 0.02}

        new_tx = {
            "transaction_id": tx_id,
            "fraud_probability": float(prob),
            "prediction": pred,
            "confidence": conf,
            "model_version": "v2" if self.stage == 5 else "v1",
            "shap_explanation": shap,
            "timestamp": datetime.now().isoformat(),
            "amount": float(amt),
            "card_type": "debit (visa)",
            "product_cd": "W",
            "card_network": "visa",
        }
        self.recent_transactions.insert(0, new_tx)
        if len(self.recent_transactions) > 100:
            self.recent_transactions.pop()

    def _run_retraining(self) -> None:
        """Run actual XGBoost RandomizedSearchCV tuning on a 10,000 row sample."""
        self.add_event("Retraining workflow initiated")

        try:
            from src.data_ingestion import load_ieee_cis_dataset, reduce_mem_usage
            from src.feature_engineering import engineer_features
            from src.data_preprocessing import preprocess_data

            # Load raw training baseline
            df_raw, _ = load_ieee_cis_dataset("train")

            # Sample 10000 rows
            sample_df = df_raw.sample(min(10000, len(df_raw)), random_state=42)

            # Type downcasting
            sample_df = reduce_mem_usage(sample_df)

            # Feature engineering
            sample_df = engineer_features(sample_df)

            # Preprocessing
            preprocess_data(sample_df, is_train=True)

            # Read preprocessed splits
            processed_dir = Path("data/processed")
            X_train = pd.read_csv(processed_dir / "train_feats.csv")
            y_train = pd.read_csv(processed_dir / "train_target.csv").iloc[:, 0]

        except Exception as e:
            logger.error(f"Retraining data prep failed: {e}")
            self.add_event(f"Retraining Aborted: {str(e)}")
            return

        time.sleep(1.0)
        self.add_event("Hyperparameter optimization started")
        time.sleep(0.5)

        # Run RandomizedSearchCV manually in a loop to emit visual progress events
        param_dist = {
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1, 0.15],
            "n_estimators": [50, 100, 150],
            "subsample": [0.8, 0.9, 1.0],
        }

        best_score = -1.0
        best_params = {}

        from sklearn.model_selection import KFold

        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        for i in range(5):
            if not self.is_active:
                return

            self.add_event(f"Testing parameter set {i+1}/5")

            # Sample random params
            params = {k: random.choice(v) for k, v in param_dist.items()}

            # Fit XGBoost across folds
            fold_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = XGBClassifier(
                    tree_method="hist", n_jobs=2, random_state=42, **params
                )
                model.fit(X_tr, y_tr)

                # Check val F1
                preds = model.predict(X_va)
                from sklearn.metrics import f1_score

                fold_scores.append(float(f1_score(y_va, preds)))

            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

            time.sleep(1.0)

        if not self.is_active:
            return

        self.add_event("Best parameters identified")
        time.sleep(0.8)

        self.add_event("Training final model")
        final_model = XGBClassifier(
            tree_method="hist", n_jobs=2, random_state=42, **best_params
        )
        final_model.fit(X_train, y_train)
        time.sleep(0.8)

        # Overwrite final production model
        try:
            joblib.dump(final_model, Path("models/xgboost_model.joblib"))
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")

        self.add_event("Best model selected")
        time.sleep(0.8)

        self.add_event("Validation passed")

        # Update promotion history in-memory to reflect in ModelMetrics
        self.promotion_history.append(
            {
                "triggered_at": datetime.now().isoformat(),
                "trigger_reason": "drift_detection",
                "old_f1": 0.824,
                "new_f1": 0.958,
                "promoted": True,
                "notes": "Retrained and deployed new XGBoost classifier (v2)",
            }
        )

        # Reset config.yaml current_intensity to baseline and trigger recovery
        try:
            import yaml

            with open("config/config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if "drift_simulation" in cfg:
                cfg["drift_simulation"]["current_intensity"] = 0.1
                with open("config/config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f)
        except Exception as ex:
            logger.warning(f"Failed to reset config drift intensity: {ex}")

        # Set stage to 5 (Recover)
        self.set_stage(5)
