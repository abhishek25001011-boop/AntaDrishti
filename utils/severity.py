"""Utility functions for assigning severity levels."""

from config import SEVERITY_LEVELS


def get_severity(score: int) -> str:
    """Map a numeric score to a severity label."""
    for label, (low, high) in SEVERITY_LEVELS.items():
        if low <= score <= high:
            return label
    return "Low"
