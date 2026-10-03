"""Experimental close-pair motion heuristic; no trained fight model is present."""

import math
from collections import deque

import cv2
import numpy as np

from config import (
    ALTERCATION_CONFIRM_HITS,
    ALTERCATION_MIN_MOTION_HITS,
    ALTERCATION_WINDOW_FRAMES,
    ALTERCATION_PROXIMITY_SCALE,
    OPTICAL_FLOW_MAGNITUDE_THRESHOLD,
    OPTICAL_FLOW_MOTION_FRACTION,
)
from detectors.person_detector import get_person_detections


class RapidInteractionHeuristic:
    """Require prolonged contact by the same pair plus repeated localized motion."""

    MIN_PERSON_HEIGHT_RATIO = 0.30
    MIN_PAIR_OVERLAP_FRACTION = 0.10
    MAX_HORIZONTAL_BOX_RATIO = 1.25
    PAIR_STABILITY_SCALE = 0.20

    def __init__(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.qualifying_frames = 0
        self.motion_frames = 0
        self.active = False
        self._recent_candidates: deque[
            tuple[tuple[tuple[float, float], tuple[float, float]], float, bool] | None
        ] = deque(maxlen=ALTERCATION_WINDOW_FRAMES)

    @classmethod
    def _same_pair(cls, first, second) -> bool:
        positions_a, scale_a = first[:2]
        positions_b, scale_b = second[:2]
        tolerance = cls.PAIR_STABILITY_SCALE * max(scale_a, scale_b)
        direct = (
            math.dist(positions_a[0], positions_b[0]) <= tolerance
            and math.dist(positions_a[1], positions_b[1]) <= tolerance
        )
        swapped = (
            math.dist(positions_a[0], positions_b[1]) <= tolerance
            and math.dist(positions_a[1], positions_b[0]) <= tolerance
        )
        return direct or swapped

    @staticmethod
    def _upper_body_motion(
        flow: np.ndarray, magnitude: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[float, tuple[float, float]] | None:
        height, width = magnitude.shape
        x1, y1, x2, y2 = map(int, box)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        body_bottom = min(y2, y1 + max(1, int((y2 - y1) * 0.72)))
        region = magnitude[y1:body_bottom, x1:x2]
        vectors = flow[y1:body_bottom, x1:x2]
        if region.size == 0:
            return None
        active = region > OPTICAL_FLOW_MAGNITUDE_THRESHOLD
        fraction = float(np.mean(active))
        if not np.any(active):
            return fraction, (0.0, 0.0)
        mean_vector = np.mean(vectors[active], axis=0)
        return fraction, (float(mean_vector[0]), float(mean_vector[1]))

    def update(self, frame: np.ndarray, people: list[dict]) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidate = None
        if self.previous_gray is not None and self.previous_gray.shape == gray.shape and len(people) >= 2:
            flow = cv2.calcOpticalFlowFarneback(
                self.previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
            frame_height, _ = gray.shape
            usable = []
            for person in people:
                box = tuple(map(int, person["box"]))
                x1, y1, x2, y2 = box
                box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
                if box_height < frame_height * self.MIN_PERSON_HEIGHT_RATIO:
                    continue
                if box_width / box_height >= self.MAX_HORIZONTAL_BOX_RATIO:
                    continue
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                motion = self._upper_body_motion(flow, magnitude, box)
                if motion is not None:
                    usable.append((center, box_height, motion[0], motion[1], box))

            for index, first in enumerate(usable):
                for second in usable[index + 1:]:
                    scale = max(first[1], second[1])
                    if math.dist(first[0], second[0]) > ALTERCATION_PROXIMITY_SCALE * scale:
                        continue
                    ax1, ay1, ax2, ay2 = first[4]
                    bx1, by1, bx2, by2 = second[4]
                    overlap = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
                    smaller_area = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
                    overlap_fraction = overlap / max(smaller_area, 1)
                    if overlap_fraction < self.MIN_PAIR_OVERLAP_FRACTION:
                        continue

                    relative_motion = math.dist(first[3], second[3])
                    motion_hit = (
                        min(first[2], second[2]) >= OPTICAL_FLOW_MOTION_FRACTION
                        and relative_motion >= OPTICAL_FLOW_MAGNITUDE_THRESHOLD
                    )
                    pair_positions = tuple(sorted((first[0], second[0]), key=lambda point: point[0]))
                    score = overlap_fraction + (1.0 if motion_hit else 0.0)
                    if candidate is None or score > candidate[0]:
                        candidate = (score, pair_positions, scale, motion_hit)

        self.previous_gray = gray
        current = None
        if candidate is not None:
            _, pair_positions, scale, motion_hit = candidate
            current = (pair_positions, scale, motion_hit)
        self._recent_candidates.append(current)

        # Repeated same-pair contact is required across most of the rolling window;
        # several frames must also contain rapid, non-shared upper-body motion.
        history = [item for item in self._recent_candidates if item is not None]
        best_contact, best_motion = max((
            (
                sum(self._same_pair(anchor, other) for other in history),
                sum(self._same_pair(anchor, other) and other[2] for other in history),
            )
            for anchor in history
        ), default=(0, 0))
        self.qualifying_frames = best_contact
        self.motion_frames = best_motion
        confirmed = (
            self.qualifying_frames >= ALTERCATION_CONFIRM_HITS
            and self.motion_frames >= ALTERCATION_MIN_MOTION_HITS
            and not self.active
        )
        if not history:
            self.active = False
        elif confirmed:
            self.active = True
        return confirmed


def detect_violence(frame: np.ndarray, detections: list[dict], tracker: RapidInteractionHeuristic) -> bool:
    """Return a heuristic rapid-interaction signal, not a fight classification."""
    people = get_person_detections(detections)
    return tracker.update(frame, people)
