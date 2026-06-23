"""SQLAlchemy models for FraudShield tables."""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
)
from sqlalchemy.sql import func

from src.api.database import Base


class FeatureRecord(Base):
    """ORM model for the features table."""

    __tablename__ = "features"

    transaction_id = Column(String, primary_key=True, index=True)
    amount = Column(Float)
    hour = Column(Integer)
    day_of_week = Column(Integer)
    is_weekend = Column(Boolean)
    user_key = Column(String)
    merchant_key = Column(String)
    user_tx_count_30d = Column(Integer)
    user_avg_amount_30d = Column(Float)
    merchant_tx_count_30d = Column(Integer)
    velocity_1hr = Column(Integer)
    card_network = Column(String)
    card_type = Column(String)
    created_at = Column(DateTime, default=func.now())


class PredictionRecord(Base):
    """ORM model for the predictions table."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(String, index=True)
    fraud_probability = Column(Float)
    prediction = Column(String)
    model_version = Column(String)
    shap_values = Column(JSON)
    timestamp = Column(DateTime, default=func.now())


class DriftLogRecord(Base):
    """ORM model for the drift_logs table."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_date = Column(Date)
    feature_name = Column(String)
    drift_score = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Boolean)
    drift_intensity = Column(Float)
    timestamp = Column(DateTime, default=func.now())


class RetrainLogRecord(Base):
    """ORM model for the retrain_logs table."""

    __tablename__ = "retrain_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    triggered_at = Column(DateTime)
    trigger_reason = Column(String)
    old_f1 = Column(Float)
    new_f1 = Column(Float)
    promoted = Column(Boolean)
    notes = Column(String)
    timestamp = Column(DateTime, default=func.now())
