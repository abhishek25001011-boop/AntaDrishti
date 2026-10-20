"""Thread-safe handoff for browser webcam inference."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Lock

import cv2
import numpy as np

from config import YOLO_CONFIDENCE_THRESHOLD
from detectors.detection_manager import DetectionManager, DetectionResult
from utils.event_detection import DetectedEvent, EventDetector

logger = logging.getLogger(__name__)


@dataclass
class WebcamUpdate:
    """The latest callback result consumed by the Streamlit script thread."""

    result: DetectionResult
    events: list[DetectedEvent]
    event_frames: list[np.ndarray] = field(default_factory=list)
    processed_at: float = field(default_factory=time.monotonic)


class WebcamProcessor:
    """Run inference in the WebRTC worker and hand results to Streamlit safely."""

    def __init__(
        self,
        manager: DetectionManager,
        conf_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        process_every_n_frames: int = 3,
        queue_size: int = 8,
    ) -> None:
        self.manager = manager
        self.conf_threshold = conf_threshold
        self.process_every_n_frames = max(1, int(process_every_n_frames))
        self._event_detector = EventDetector()
        self._updates: Queue[WebcamUpdate] = Queue(maxsize=max(1, int(queue_size)))
        self._frame_index = 0
        self._lock = Lock()

    def callback(self, video_frame):
        """Annotate frames without touching Streamlit, SQLite, or session state."""
        try:
            frame = video_frame.to_ndarray(format="bgr24")
            if (
                not isinstance(frame, np.ndarray)
                or frame.ndim != 3
                or frame.shape[2] != 3
                or frame.size == 0
            ):
                logger.warning("Skipping invalid browser webcam frame")
                return video_frame

            with self._lock:
                self._frame_index += 1
                if self._frame_index % self.process_every_n_frames:
                    return video_frame
                result = self.manager.process_frame(frame, self.conf_threshold)
                events = self._event_detector.process_frame(frame, result.detections)
                output = self._to_video_frame(video_frame, result.frame)
                self._publish(WebcamUpdate(
                    result=result,
                    events=events,
                    # Events from one processed frame share its immutable image.
                    event_frames=[result.frame] * len(events),
                ))
                return output
        except Exception:
            logger.exception("Browser webcam frame processing failed; passing through input frame")
            return video_frame

    def drain(self) -> list[WebcamUpdate]:
        """Return queued updates in the Streamlit script thread."""
        updates: list[WebcamUpdate] = []
        while True:
            try:
                updates.append(self._updates.get_nowait())
            except Empty:
                return updates

    def reset(self) -> None:
        """Discard stale callback results when a camera session is restarted."""
        with self._lock:
            self._frame_index = 0
            self._event_detector = EventDetector()
            while True:
                try:
                    self._updates.get_nowait()
                except Empty:
                    return

    def _publish(self, update: WebcamUpdate) -> None:
        while len(update.event_frames) < len(update.events):
            update.event_frames.append(update.result.frame)
        try:
            self._updates.put_nowait(update)
        except Full:
            try:
                displaced = self._updates.get_nowait()
            except Empty:
                displaced = None
            if displaced is not None and displaced.events:
                # Keep event handoffs when coalescing stale video updates. One
                # event per type is enough because persistence cooldowns are
                # keyed by event type and source.
                retained = {event.event_type for event in update.events}
                for index, event in enumerate(displaced.events):
                    if event.event_type in retained:
                        continue
                    update.events.insert(0, event)
                    frame = (
                        displaced.event_frames[index]
                        if index < len(displaced.event_frames)
                        else displaced.result.frame
                    )
                    update.event_frames.insert(0, frame)
                    retained.add(event.event_type)
            try:
                self._updates.put_nowait(update)
            except Full:
                pass

    @staticmethod
    def _to_video_frame(video_frame, frame: np.ndarray):
        """Preserve the component's frame type without importing WebRTC internals."""
        return video_frame.from_ndarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), format="rgb24")
