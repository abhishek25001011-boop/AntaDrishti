"""Singleton detection manager for loading and reusing the YOLO model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2

from detectors.yolo_detector import BAG_CLASSES, TARGET_CLASSES, load_yolo_model


@dataclass
class DetectionResult:
    """Structured output for a processed frame."""

    frame: Any
    detections: list[dict]
    person_count: int
    bag_count: int
    fps: float
    crowd_density: str


class DetectionManager:
    """Load the YOLOv8 model once and reuse it for every frame."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._class_names = None
        return cls._instance

    def get_model(self):
        """Load the detector model on demand and cache it."""
        if self._model is None:
            self._model = load_yolo_model()
            self._class_names = self._model.names
        return self._model

    @staticmethod
    def crowd_density(person_count: int) -> str:
        """Map person count to a simple crowd density label."""
        if person_count < 10:
            return "LOW"
        if person_count <= 20:
            return "MEDIUM"
        return "HIGH"

    def process_frame(self, frame, conf_threshold: float = 0.35) -> DetectionResult:
        """Run detection on one frame and annotate it with summary metrics."""
        start_time = time.perf_counter()
        model = self.get_model()
        results = model(frame, device="cpu", conf=conf_threshold, verbose=False)

        output_frame = frame.copy()
        detections: list[dict] = []
        person_count = 0
        bag_count = 0

        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for coords, confidence, class_id in zip(xyxy, confidences, classes):
                    label = self._class_names.get(class_id, str(class_id))
                    if label not in TARGET_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, coords)
                    detections.append(
                        {
                            "label": label,
                            "confidence": float(confidence),
                            "box": (x1, y1, x2, y2),
                        }
                    )

                    color = (0, 200, 0) if label == "person" else (0, 120, 255)
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        output_frame,
                        f"{label} {confidence:.2f}",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    if label == "person":
                        person_count += 1
                    elif label in BAG_CLASSES:
                        bag_count += 1

        fps = 1.0 / max(time.perf_counter() - start_time, 1e-6)
        crowd_density = self.crowd_density(person_count)

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
        )