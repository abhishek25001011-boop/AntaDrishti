"""In-memory per-source event cooldown to limit duplicate alerts and snapshots."""

import time

from config import ALERT_COOLDOWN_SECONDS


class AlertCooldown:
    def __init__(self, cooldown_seconds: float = ALERT_COOLDOWN_SECONDS) -> None:
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._last_alert: dict[tuple[str, str], float] = {}

    def allow(self, event_type: str, source: str, now: float | None = None) -> bool:
        """Return true once per event/source during the configured time window."""
        now = time.monotonic() if now is None else now
        key = (str(source), str(event_type))
        last = self._last_alert.get(key)
        if last is not None and now - last < self.cooldown_seconds:
            return False
        self.mark(event_type, source, now)
        return True

    def mark(self, event_type: str, source: str, now: float | None = None) -> None:
        """Start the cooldown after an incident has been persisted successfully."""
        now = time.monotonic() if now is None else now
        self._last_alert[(str(source), str(event_type))] = now

    def forget(self, event_type: str, source: str) -> None:
        """Release a cooldown reservation when persistence did not succeed."""
        self._last_alert.pop((str(source), str(event_type)), None)
