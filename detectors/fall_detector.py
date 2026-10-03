"""Fall posture heuristic; needs temporal person detections and is not a model."""

from detectors.person_detector import get_person_detections


def detect_fall(detections: list[dict], tracker, now: float | None = None) -> list:
    """Update the caller's temporal tracker and return newly confirmed falls."""
    people = get_person_detections(detections)
    return tracker.update_fall_tracks(people, now=now)
