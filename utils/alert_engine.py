"""Utility helpers for building alert messages.

This module keeps alert formatting simple and beginner-friendly.
"""


def build_alert_message(alert_type, severity, message):
    """Create a human-readable alert message."""
    return {
        "type": alert_type,
        "severity": severity,
        "message": message,
    }


def build_alert_record(event: str, severity: str, message: str, status: str = "Active"):
    """Create a standardized alert record for database storage."""
    payload = build_alert_message(event, severity, message)
    payload["status"] = status
    return payload


def record_incident(frame, event_type: str, severity: str, message: str, source: str) -> dict:
    """Save an incident snapshot and its corresponding SQLite alert row."""
    from config import SNAPSHOT_DIR
    from database.database import save_alert
    from utils.snapshot import save_snapshot

    slug = "".join(char.lower() if char.isalnum() else "_" for char in event_type).strip("_")[:32]
    snapshot_path = save_snapshot(frame, SNAPSHOT_DIR, prefix=slug or "incident")
    alert_id = save_alert(
        event_type, severity, message, source=source, snapshot_path=snapshot_path
    )
    return {
        "id": alert_id,
        "event_type": event_type,
        "severity": severity,
        "message": message,
        "source": source,
        "snapshot_path": snapshot_path,
    }
