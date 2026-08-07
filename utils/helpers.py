"""General helper functions for the AntaDrishti prototype."""


def format_alert_count(value: int) -> str:
    """Format alert counts for display."""
    return str(value)


def safe_int(value, default=0) -> int:
    """Safely convert a value to an integer."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
