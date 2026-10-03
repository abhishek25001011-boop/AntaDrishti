"""Singleton detection manager for loading and reusing the YOLO model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2

from config import YOLO_CONFIDENCE_THRESHOLD
from detectors.yolo_detector import load_yolo_model, run_detection
from detectors.crowd_detector import measure_crowd


@dataclass
class DetectionResult:
    """Structured output for a processed frame."""

    frame: Any
    detections: list[dict]
    person_count: int
    bag_count: int
    fps: float
    crowd_density: str
    crowd_occupancy: float


class DetectionManager:
    """Load the YOLOv8 model once and reuse it for every frame."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def get_model(self):
        """Load the detector model on demand and cache it."""
        if self._model is None:
            self._model = load_yolo_model()
        return self._model

    def process_frame(self, frame, conf_threshold: float = YOLO_CONFIDENCE_THRESHOLD) -> DetectionResult:
        """Run detection on one frame and annotate it with summary metrics."""
        start_time = time.perf_counter()
        model = self.get_model()
        output_frame, _, bag_count, detections = run_detection(frame, model, conf_threshold)

        fps = 1.0 / max(time.perf_counter() - start_time, 1e-6)
        crowd_occupancy, person_count, crowd_density = measure_crowd(detections, frame.shape)

        overlay_lines = [
            f"FPS: {fps:.2f}",
            f"Person Count: {person_count}",
            f"Crowd Density: {crowd_density}",
        ]
        if bag_count:
            overlay_lines.append(f"Bag Count: {bag_count}")

        for index, text in enumerate(overlay_lines):
            cv2.putText(
                output_frame,
                text,
                (12, 28 + (index * 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return DetectionResult(
            frame=output_frame,
            detections=detections,
            person_count=person_count,
            bag_count=bag_count,
            fps=fps,
            crowd_density=crowd_density,
            crowd_occupancy=crowd_occupancy,
        )
