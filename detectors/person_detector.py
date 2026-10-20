"""Person-class filtering shared by crowd and event assessment."""

import math


def is_valid_box(box) -> bool:
    """Return whether a detection has a finite, positive-area xyxy box."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        all(math.isfinite(value) for value in (x1, y1, x2, y2))
        and x2 > x1
        and y2 > y1
    )


def get_person_detections(detections: list[dict]) -> list[dict]:
    """Return only person boxes from real object-detector outputs."""
    if not isinstance(detections, (list, tuple)):
        return []
    return [
        item
        for item in detections
        if isinstance(item, dict)
        and item.get("label") == "person"
        and is_valid_box(item.get("box"))
    ]
