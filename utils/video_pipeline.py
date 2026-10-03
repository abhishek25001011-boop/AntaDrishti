"""Bounded frame processing shared by sample, upload, and camera inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2

from config import MAX_PROCESSING_FRAMES, YOLO_CONFIDENCE_THRESHOLD
from detectors.detection_manager import DetectionManager, DetectionResult
from utils.alert_cooldown import AlertCooldown
from utils.alert_engine import record_incident
from utils.event_detection import EventDetector


@dataclass
class PipelineSummary:
    frames_processed: int = 0
    frame_limit: int = MAX_PROCESSING_FRAMES
    total_frames: int | None = None
    reached_end: bool = False
    stopped: bool = False
    error: str | None = None
    incidents: list[dict] = field(default_factory=list)


FrameCallback = Callable[[int, int, int | None, DetectionResult, list[dict]], None]


def process_capture(
    capture,
    source: str,
    manager: DetectionManager,
    event_detector: EventDetector,
    cooldown: AlertCooldown,
    conf_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    frame_limit: int = MAX_PROCESSING_FRAMES,
    on_frame: FrameCallback | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> PipelineSummary:
    """Process at most ``frame_limit`` frames and return control to the caller."""
    frame_limit = max(1, int(frame_limit))
    raw_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total = raw_total if raw_total > 0 else None
    summary = PipelineSummary(frame_limit=frame_limit, total_frames=total)

    while summary.frames_processed < frame_limit:
        if should_stop and should_stop():
            summary.stopped = True
            break
        ok, frame = capture.read()
        if not ok:
            summary.reached_end = True
            if summary.frames_processed == 0:
                summary.error = "The video opened but did not provide a readable frame."
            break

        result = manager.process_frame(frame, conf_threshold)
        events = event_detector.process_frame(frame, result.detections)
        frame_incidents = []
        for event in events:
            if not cooldown.allow(event.event_type, source):
                continue
            incident = record_incident(
                frame, event.event_type, event.severity, event.message, source
            )
            summary.incidents.append(incident)
            frame_incidents.append(incident)
            if event.box:
                x1, y1, x2, y2 = event.box
                cv2.rectangle(result.frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                result.frame, event.event_type[:48],
                (12, result.frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 2, cv2.LINE_AA,
            )

        summary.frames_processed += 1
        if on_frame:
            on_frame(summary.frames_processed, frame_limit, total, result, frame_incidents)

    if total is not None and total <= frame_limit and summary.frames_processed >= total:
        summary.reached_end = True
    return summary
