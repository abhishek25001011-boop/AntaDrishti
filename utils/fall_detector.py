"""Lightweight fall detection logic based on YOLO person bounding boxes."""

import cv2
from typing import Any, Dict, List, Tuple

from config import SNAPSHOT_DIR
from database.database import save_alert
from utils.alert_engine import build_alert_record

FALL_DURATION_SECONDS = 3.0


def is_fall_box(box: Tuple[int, int, int, int]) -> bool:
    """Return True when a person bounding box is wider than it is tall."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return width > height


def save_fall_snapshot(frame: Any, alert_id: int) -> str:
    """Save one fall snapshot to the snapshots directory."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"fall_alert_{alert_id}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def create_fall_alert(frame: Any, message: str) -> int:
    """Store a new fall alert in SQLite and save a snapshot."""
    alert_record = build_alert_record("Fall Detection", "Critical", message, status="Active")
    alert_id = save_alert(
        alert_record["type"],
        alert_record["severity"],
        alert_record["message"],
        status=alert_record["status"],
    )
    save_fall_snapshot(frame, alert_id)
    return alert_id


def annotate_fall(frame: Any, box: Tuple[int, int, int, int], message: str) -> Any:
    """Draw a red fall bounding box and overlay a critical message."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        frame,
        message,
        (x1, max(y1 - 24, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def find_fallen_person(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return detections that meet the fall geometry condition."""
    return [item for item in detections if item["label"] == "person" and is_fall_box(item["box"])]
