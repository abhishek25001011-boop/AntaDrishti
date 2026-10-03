"""SQLite helpers for storing and querying incident alerts."""

import sqlite3
from contextlib import closing

from config import DB_PATH

_initialized_path = None


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create the alert table and add new fields to existing databases."""
    global _initialized_path
    if _initialized_path == DB_PATH:
        return
    with closing(_connect()) as conn, conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL DEFAULT 'Unknown',
                severity TEXT NOT NULL DEFAULT 'Low',
                message TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                snapshot_path TEXT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'Active',
                alert_type TEXT,
                created_at TEXT
            )"""
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(alerts)")}
        migrations = {
            "event_type": "TEXT NOT NULL DEFAULT 'Unknown'",
            "source": "TEXT NOT NULL DEFAULT ''",
            "snapshot_path": "TEXT",
            "timestamp": "TEXT",
        }
        for column, definition in migrations.items():
            if column not in columns:
                conn.execute(f"ALTER TABLE alerts ADD COLUMN {column} {definition}")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(alerts)")}
        if "alert_type" in columns:
            conn.execute("UPDATE alerts SET event_type = alert_type WHERE (event_type IS NULL OR event_type = 'Unknown') AND alert_type IS NOT NULL")
        if "created_at" in columns:
            conn.execute("UPDATE alerts SET timestamp = created_at WHERE timestamp IS NULL AND created_at IS NOT NULL")
        conn.execute("UPDATE alerts SET timestamp = CURRENT_TIMESTAMP WHERE timestamp IS NULL")
        conn.execute("UPDATE alerts SET severity = CASE lower(severity) WHEN 'critical' THEN 'High' WHEN 'high' THEN 'High' WHEN 'medium' THEN 'Medium' ELSE 'Low' END")
    _initialized_path = DB_PATH


def save_alert(
    alert_type: str,
    severity: str,
    message: str,
    status: str = "Active",
    source: str = "",
    snapshot_path: str | None = None,
    timestamp: str | None = None,
) -> int:
    """Save one alert while keeping the legacy column names populated."""
    normalized_severity = {"critical": "High", "high": "High", "medium": "Medium", "low": "Low"}.get(
        str(severity).strip().lower(), "Low"
    )
    init_db()
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            """INSERT INTO alerts
            (event_type, alert_type, severity, message, source, snapshot_path, timestamp, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), COALESCE(?, CURRENT_TIMESTAMP), ?)""",
            (alert_type, alert_type, normalized_severity, message, source, snapshot_path, timestamp, timestamp, status),
        )
        return int(cursor.lastrowid)


def count_alerts(status: str | None = None, severity: str | None = None) -> int:
    """Return the count of alerts matching optional filters."""
    init_db()
    query = "SELECT COUNT(*) FROM alerts"
    conditions, params = [], []
    if status is not None:
        conditions.append("status = ?")
        params.append(status)
    if severity is not None:
        severity = {"critical": "High"}.get(severity.lower(), severity)
        conditions.append("severity = ?")
        params.append(severity)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    with closing(_connect()) as conn, conn:
        return int(conn.execute(query, params).fetchone()[0])


def fetch_alerts(limit: int | None = 10):
    """Return recent alerts, or the full incident history when limit is None."""
    init_db()
    with closing(_connect()) as conn, conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM alerts ORDER BY timestamp DESC, id DESC"
        if limit is not None:
            return conn.execute(query + " LIMIT ?", (max(0, int(limit)),)).fetchall()
        return conn.execute(query).fetchall()


def count_alerts_last_hours(hours: int = 24) -> int:
    """Count real incident rows inside a recent time window."""
    init_db()
    modifier = f"-{max(1, int(hours))} hours"
    with closing(_connect()) as conn, conn:
        return int(conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE datetime(timestamp) >= datetime('now', ?)",
            (modifier,),
        ).fetchone()[0])
