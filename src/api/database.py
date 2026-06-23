"""Database connection and session helpers for FastAPI."""

from __future__ import annotations

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")
engine = None
SessionLocal = None

if DATABASE_URL:
    try:
        # Enable connection pooling and pre-ping to detect dropped connections
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as e:
        print(f"Warning: Database connection failed initialization: {e}")
        engine = None
        SessionLocal = None

Base = declarative_base()


def get_db():
    """Dependency to get database session with fallback handling."""
    if SessionLocal is None:
        yield None
        return

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_health() -> bool:
    """Check if the database is online and reachable."""
    if engine is None:
        return False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
