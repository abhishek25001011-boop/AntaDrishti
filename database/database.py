"""SQLite database helpers for storing alert events."""

import sqlite3
from pathlib import Path

from config import DB_PATH


def init_db() -> None:
    """Create the database and required tables if they do not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Active',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def save_alert(alert_type: str, severity: str, message: str, status: str = "Active") -> int:
    """Save one alert record to the SQLite database and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO alerts (alert_type, severity, message, status) VALUES (?, ?, ?, ?)",
        (alert_type, severity, message, status),
    )
    alert_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return alert_id


def count_alerts(status: str = None, severity: str = None) -> int:
    """Return the count of alerts matching optional status or severity filters."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = "SELECT COUNT(*) FROM alerts"
    params = []
    conditions = []

    if status is not None:
        conditions.append("status = ?")
        params.append(status)
    if severity is not None:
        conditions.append("severity = ?")
        params.append(severity)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    cursor.execute(query, params)
    count = cursor.fetchone()[0]
    conn.close()
    return count


def fetch_alerts(limit: int = 10):
    """Return recent alerts from the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows
