"""Temporal, explicitly heuristic event checks for YOLO person detections."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from detectors.crowd_detector import measure_crowd
from detectors.fall_detector import detect_fall
from detectors.person_detector import get_person_detections
from detectors.yolo_detector import BAG_CLASSES
from detectors.violence_detector import RapidInteractionHeuristic, detect_violence
from config import (
    CROWD_CLEAR_SECONDS,
    CROWD_COUNT_THRESHOLD,
    CROWD_HIGH_COUNT_THRESHOLD,
    CROWD_HIGH_OCCUPANCY_THRESHOLD,
    CROWD_OCCUPANCY_THRESHOLD,
    FALL_PERSISTENCE_SECONDS,
    FALL_TRACK_TIMEOUT_SECONDS,
)


@dataclass
class DetectedEvent:
    event_type: str
    severity: str
    message: str
    box: tuple[int, int, int, int] | None = None


class EventDetector:
    """Track conservative fall, occupancy, and motion heuristics across frames."""

    def __init__(self) -> None:
        self._tracks: list[dict] = []
        self._crowd_active = False
        self._crowd_severity: str | None = None
        self._crowd_clear_since: float | None = None
        self._motion_heuristic = RapidInteractionHeuristic()

    def process_frame(
        self, frame: np.ndarray, detections: list[dict], now: float | None = None
    ) -> list[DetectedEvent]:
        """Return newly confirmed events; callers decide how to persist them."""
        now = time.monotonic() if now is None else now
        persons = get_person_detections(detections)
        events = detect_fall(persons, self, now)
        events.extend(self._bag_events(detections))
        events.extend(self._crowd_events(persons, frame.shape, now))
        events.extend(self._motion_events(frame, persons))
        return events

    def update_fall_tracks(self, persons: list[dict], now: float | None = None) -> list[DetectedEvent]:
        """Advance posture tracks; falls require upright-to-horizontal persistence."""
        now = time.monotonic() if now is None else now
        events: list[DetectedEvent] = []
        unmatched = set(range(len(self._tracks)))
        next_tracks = []

        for person in persons:
            box = tuple(map(int, person["box"]))
            x1, y1, x2, y2 = box
            width, height = max(1, x2 - x1), max(1, y2 - y1)
            horizontal = width / height >= 1.2 and width >= 40
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            best_index, best_distance = None, float("inf")
            for index in unmatched:
                track = self._tracks[index]
                if now - track.get("last_seen", now) > FALL_TRACK_TIMEOUT_SECONDS:
                    continue
                tx1, ty1, tx2, ty2 = track["box"]
                track_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
                distance = math.dist(center, track_center)
                scale = max(width, height, tx2 - tx1, ty2 - ty1)
                if distance < max(35, scale * 0.75) and distance < best_distance:
                    best_index, best_distance = index, distance

            if best_index is None:
                track = {"box": box, "upright_seen": not horizontal, "horizontal_since": now if horizontal else None, "alerted": False}
            else:
                unmatched.remove(best_index)
                track = self._tracks[best_index]
                if not horizontal:
                    track["upright_seen"] = True
                    track["horizontal_since"] = None
                    track["alerted"] = False
                elif track["horizontal_since"] is None:
                    track["horizontal_since"] = now
                elif (
                    track["upright_seen"]
                    and not track["alerted"]
                    and now - track["horizontal_since"] >= FALL_PERSISTENCE_SECONDS
                ):
                    track["alerted"] = True
                    events.append(DetectedEvent(
                        "Possible Fall (posture heuristic)", "High",
                        "Person changed from upright to a sustained horizontal posture; review required.", box,
                    ))
                track["box"] = box
            track["last_seen"] = now
            next_tracks.append(track)

        for index in unmatched:
            track = self._tracks[index]
            if now - track.get("last_seen", now) <= FALL_TRACK_TIMEOUT_SECONDS:
                next_tracks.append(track)
        self._tracks = next_tracks
        return events

    def _bag_events(self, detections: list[dict]) -> list[DetectedEvent]:
        """Turn valid YOLO bag-class detections into ordinary pipeline incidents."""
        events = []
        for detection in detections:
            label = detection.get("label")
            box = detection.get("box")
            if label not in BAG_CLASSES or box is None:
                continue
            confidence = float(detection.get("confidence", 0.0))
            events.append(DetectedEvent(
                "Bag Detection", "Low",
                f"YOLO detected a {label} ({confidence:.0%} confidence); review required.",
                tuple(map(int, box)),
            ))
        return events

    def _crowd_events(
        self, persons: list[dict], frame_shape: tuple[int, ...], now: float
    ) -> list[DetectedEvent]:
        occupancy, count, _ = measure_crowd(persons, frame_shape)
        crowded = count >= CROWD_COUNT_THRESHOLD and occupancy >= CROWD_OCCUPANCY_THRESHOLD
        if not crowded:
            if self._crowd_active:
                if self._crowd_clear_since is None:
                    self._crowd_clear_since = now
                elif now - self._crowd_clear_since >= self.CROWD_CLEAR_SECONDS:
                    self._crowd_active = False
                    self._crowd_severity = None
                    self._crowd_clear_since = None
            return []

        self._crowd_clear_since = None
        severity = "High" if count >= CROWD_HIGH_COUNT_THRESHOLD and occupancy >= CROWD_HIGH_OCCUPANCY_THRESHOLD else "Medium"
        if self._crowd_active and not (severity == "High" and self._crowd_severity == "Medium"):
            return []
        self._crowd_active = True
        self._crowd_severity = severity
        return [DetectedEvent(
            "Crowd Density", severity,
            f"{count} detected people occupy about {occupancy:.0%} of the frame.",
        )]

    def _motion_events(self, frame: np.ndarray, persons: list[dict]) -> list[DetectedEvent]:
        if detect_violence(frame, persons, self._motion_heuristic):
            return [DetectedEvent(
                "Possible Physical Altercation (motion heuristic)", "Medium",
                "Two nearby detected people showed repeated rapid motion; this is not an AI fight classification.",
            )]
        return []
