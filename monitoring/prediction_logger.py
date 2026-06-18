"""
prediction_logger.py — Structured prediction logging (SQLite)
French-Learning-Perceptions ML — MLOps Level 3

Usage:
    from monitoring.prediction_logger import init_predictions_db, log_prediction

    db_path = Path("monitoring/predictions.db")
    init_predictions_db(db_path)
    log_prediction(db_path, "h1", {"nb_langues": 3}, "Oui", 0.87, 12.5)
"""
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    model_key     TEXT    NOT NULL,
    model_version TEXT    NOT NULL DEFAULT 'latest',
    input_json    TEXT    NOT NULL,
    prediction    TEXT    NOT NULL,
    probability   REAL,
    latency_ms    REAL    NOT NULL,
    drift_batch   TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_model_time
    ON predictions(model_key, timestamp DESC);
"""


def get_db_path(config: Optional[dict] = None) -> Path:
    """Read db path from config or fall back to default."""
    if config:
        p = config.get("monitoring", {}).get("prediction_db", "")
        if p:
            return Path(p)
    return Path("monitoring/predictions.db")


def init_predictions_db(db_path: Path) -> None:
    """Create predictions table if it does not exist. Idempotent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        conn.commit()
        log.info("Prediction DB ready: %s", db_path)
    except sqlite3.DatabaseError as e:
        log.error("Failed to init predictions DB: %s", e)
        raise
    finally:
        conn.close()


def log_prediction(
    db_path: Path,
    model_key: str,
    model_version: str,
    input_data: dict,
    prediction: str | float | int,
    probability: Optional[float] = None,
    latency_ms: Optional[float] = None,
    drift_batch: Optional[str] = None,
) -> int:
    """Insert one prediction row. Returns the new row id."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            """INSERT INTO predictions
               (timestamp, model_key, model_version, input_json,
                prediction, probability, latency_ms, drift_batch)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                model_key,
                model_version,
                json.dumps(input_data, ensure_ascii=False, default=str),
                str(prediction),
                probability,
                latency_ms or 0.0,
                drift_batch,
            ),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.DatabaseError as e:
        log.warning("Prediction log failed for %s: %s", model_key, e)
        return -1
    finally:
        conn.close()


def get_recent_predictions(
    db_path: Path,
    model_key: str,
    limit: int = 500,
    since: Optional[datetime] = None,
) -> pd.DataFrame:
    """Fetch recent predictions for drift analysis."""
    query = "SELECT timestamp, input_json, prediction, probability FROM predictions WHERE model_key = ?"
    params = [model_key]
    if since:
        query += " AND timestamp >= ?"
        params.append(since.isoformat())
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def get_prediction_count(
    db_path: Path,
    model_key: Optional[str] = None,
    since: Optional[datetime] = None,
) -> int:
    """Count predictions, optionally filtered."""
    query = "SELECT COUNT(*) FROM predictions WHERE 1=1"
    params = []
    if model_key:
        query += " AND model_key = ?"
        params.append(model_key)
    if since:
        query += " AND timestamp >= ?"
        params.append(since.isoformat())
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(query, params).fetchone()[0]
    finally:
        conn.close()


def export_to_csv(db_path: Path, output_path: Path) -> Path:
    """Export all predictions to CSV for research reproducibility."""
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp", conn)
        df.to_csv(output_path, index=False)
        log.info("Exported %d predictions to %s", len(df), output_path)
        return output_path
    finally:
        conn.close()
