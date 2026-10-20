"""Unit tests for the thread-safe browser webcam handoff."""

import unittest

import numpy as np

from detectors.detection_manager import DetectionResult
from utils.event_detection import DetectedEvent
from utils.webcam_pipeline import WebcamProcessor, WebcamUpdate


class _FakeVideoFrame:
    def __init__(self, frame, format=None):
        self.frame = frame
        self.format = format

    def to_ndarray(self, format):
        self.format = format
        return self.frame

    @classmethod
    def from_ndarray(cls, frame, format):
        return cls(frame, format)


class _FakeManager:
    def __init__(self):
        self.calls = 0

    def process_frame(self, frame, confidence):
        self.calls += 1
        return DetectionResult(
            frame=frame.copy(),
            detections=[],
            person_count=0,
            bag_count=0,
            fps=10.0,
            crowd_density="LOW",
            crowd_occupancy=0.0,
        )


class WebcamPipelineTests(unittest.TestCase):
    def test_callback_only_publishes_updates_after_frame_skip(self):
        processor = WebcamProcessor(_FakeManager(), process_every_n_frames=2)
        frame = np.zeros((12, 16, 3), dtype=np.uint8)

        first = processor.callback(_FakeVideoFrame(frame))
        self.assertIsInstance(first, _FakeVideoFrame)
        self.assertEqual(processor.drain(), [])

        processor.callback(_FakeVideoFrame(frame))
        updates = processor.drain()
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].result.person_count, 0)

    def test_processed_frame_keeps_dimensions_and_converts_bgr_to_rgb(self):
        manager = _FakeManager()
        processor = WebcamProcessor(manager, process_every_n_frames=1)
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[0, 0] = (10, 20, 30)

        output = processor.callback(_FakeVideoFrame(frame))

        self.assertEqual(output.frame.shape, (12, 16, 3))
        self.assertEqual(output.frame[0, 0].tolist(), [30, 20, 10])
        self.assertEqual(output.format, "rgb24")

    def test_invalid_and_empty_frames_are_skipped(self):
        for frame in (np.empty((0, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)):
            with self.subTest(shape=frame.shape):
                manager = _FakeManager()
                processor = WebcamProcessor(manager, process_every_n_frames=1)
                video_frame = _FakeVideoFrame(frame)

                self.assertIs(processor.callback(video_frame), video_frame)
                self.assertEqual(manager.calls, 0)
                self.assertEqual(processor.drain(), [])

    def test_detection_error_does_not_escape_or_publish_partial_update(self):
        class _FailingManager:
            def process_frame(self, frame, confidence):
                raise RuntimeError("inference failed")

        processor = WebcamProcessor(_FailingManager(), process_every_n_frames=1)
        video_frame = _FakeVideoFrame(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertIs(processor.callback(video_frame), video_frame)
        self.assertEqual(processor.drain(), [])

    def test_callback_does_not_write_events_to_storage(self):
        processor = WebcamProcessor(_FakeManager(), process_every_n_frames=1)
        processor._event_detector.process_frame = lambda frame, detections: [
            DetectedEvent("Test Event", "Low", "test")
        ]

        processor.callback(_FakeVideoFrame(np.zeros((8, 8, 3), dtype=np.uint8)))

        updates = processor.drain()
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].events[0].event_type, "Test Event")

    def test_reset_discards_queued_updates(self):
        processor = WebcamProcessor(_FakeManager(), process_every_n_frames=1)
        processor.callback(_FakeVideoFrame(np.zeros((8, 8, 3), dtype=np.uint8)))
        processor.reset()
        self.assertEqual(processor.drain(), [])

    def test_full_bounded_queue_keeps_displaced_event_and_its_frame(self):
        processor = WebcamProcessor(_FakeManager(), process_every_n_frames=1, queue_size=1)
        event_frame = np.full((8, 8, 3), 17, dtype=np.uint8)
        latest_frame = np.full((8, 8, 3), 29, dtype=np.uint8)
        event = DetectedEvent("Bag Detection", "Low", "review", (1, 1, 5, 5))
        processor._publish(WebcamUpdate(
            DetectionResult(event_frame, [], 0, 0, 10.0, "LOW", 0.0),
            [event],
            [event_frame.copy()],
        ))
        processor._publish(WebcamUpdate(
            DetectionResult(latest_frame, [], 0, 0, 10.0, "LOW", 0.0),
            [],
        ))

        updates = processor.drain()
        self.assertEqual(processor._updates.maxsize, 1)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].result.frame[0, 0, 0], 29)
        self.assertEqual([item.event_type for item in updates[0].events], ["Bag Detection"])
        self.assertEqual(updates[0].event_frames[0][0, 0, 0], 17)


if __name__ == "__main__":
    unittest.main()
