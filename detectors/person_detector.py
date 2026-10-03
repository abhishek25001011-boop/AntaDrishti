"""Person-class filtering shared by crowd and event assessment."""


def get_person_detections(detections: list[dict]) -> list[dict]:
    """Return only person boxes from real object-detector outputs."""
    return [item for item in detections if item.get("label") == "person"]
