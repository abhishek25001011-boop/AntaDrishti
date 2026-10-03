"""Frame-relative crowd estimates based on real YOLO person boxes."""

from config import CROWD_COUNT_THRESHOLD, CROWD_OCCUPANCY_THRESHOLD
from detectors.person_detector import get_person_detections


def measure_crowd(detections: list[dict], frame_shape: tuple[int, ...]) -> tuple[float, int, str]:
    """Return estimated box occupancy, people count, and coarse density label."""
    height, width = frame_shape[:2]
    area = max(1, height * width)
    people = get_person_detections(detections)
    occupied = sum(
        max(0, item["box"][2] - item["box"][0]) * max(0, item["box"][3] - item["box"][1])
        for item in people
    )
    ratio = min(1.0, occupied / area)
    count = len(people)
    if count < CROWD_COUNT_THRESHOLD or ratio < CROWD_OCCUPANCY_THRESHOLD:
        density = "LOW"
    elif count < 10 or ratio < 0.25:
        density = "MEDIUM"
    else:
        density = "HIGH"
    return ratio, count, density


def detect_crowd(frame, detections: list[dict]) -> dict:
    """Describe crowd level from frame-relative detected-person occupancy."""
    occupancy, count, density = measure_crowd(detections, frame.shape)
    return {"person_count": count, "occupancy": occupancy, "density": density}
