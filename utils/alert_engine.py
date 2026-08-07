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
