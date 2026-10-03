"""Fast isolated tests for event thresholds and persistence helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import database.database as database
from detectors.crowd_detector import detect_crowd
from utils.alert_cooldown import AlertCooldown
from utils.event_detection import EventDetector
from utils.severity import get_severity
from utils.snapshot import save_snapshot


class PipelineUnitTests(unittest.TestCase):
    def test_severity_mapping(self):
        self.assertEqual(get_severity(1), "Low")
        self.assertEqual(get_severity(5), "Medium")
        self.assertEqual(get_severity(8), "High")

    def test_crowd_uses_count_and_occupancy(self):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        few = [{"label": "person", "box": (i * 100, 0, i * 100 + 50, 50)} for i in range(2)]
        self.assertEqual(detect_crowd(frame, few)["density"], "LOW")
        group = [{"label": "person", "box": (i * 78, 0, i * 78 + 70, 70)} for i in range(5)]
        result = detect_crowd(frame, group)
        self.assertEqual(result["person_count"], 5)
        self.assertGreaterEqual(result["occupancy"], 0.12)
        self.assertEqual(result["density"], "MEDIUM")

    def test_fall_requires_upright_to_persistent_horizontal_posture(self):
        detector = EventDetector()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        upright = [{"label": "person", "box": (80, 20, 120, 160)}]
        horizontal = [{"label": "person", "box": (40, 70, 160, 110)}]
        self.assertEqual(detector.process_frame(frame, upright, now=0.0), [])
        self.assertEqual(detector.process_frame(frame, horizontal, now=0.1), [])
        confirmed = detector.process_frame(frame, horizontal, now=1.0)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].severity, "High")
        self.assertIn("heuristic", confirmed[0].event_type)

    def test_bag_detection_becomes_standard_incident_event(self):
        detector = EventDetector()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = [{"label": "backpack", "confidence": 0.91, "box": (30, 40, 90, 120)}]
        events = detector.process_frame(frame, detections, now=0.0)
        bag_events = [event for event in events if event.event_type == "Bag Detection"]
        self.assertEqual(len(bag_events), 1)
        self.assertEqual(bag_events[0].severity, "Low")
        self.assertEqual(bag_events[0].box, (30, 40, 90, 120))

    def test_alert_cooldown_is_per_event_and_source(self):
        cooldown = AlertCooldown(cooldown_seconds=30)
        self.assertTrue(cooldown.allow("Crowd Density", "cam1", now=100))
        self.assertFalse(cooldown.allow("Crowd Density", "cam1", now=120))
        self.assertTrue(cooldown.allow("Crowd Density", "cam2", now=120))
        self.assertTrue(cooldown.allow("Crowd Density", "cam1", now=130))

    def test_snapshot_saves_a_nonempty_image(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(save_snapshot(np.zeros((20, 30, 3), dtype=np.uint8), folder, "unit"))
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_sqlite_insert_in_isolated_database(self):
        old_path = database.DB_PATH
        old_initialized_path = database._initialized_path
        try:
            with tempfile.TemporaryDirectory() as folder:
                database.DB_PATH = Path(folder) / "test.db"
                database._initialized_path = None
                database.init_db()
                alert_id = database.save_alert(
                    "Crowd Density", "Medium", "unit test only", source="test source"
                )
                row = dict(database.fetch_alerts(1)[0])
                self.assertEqual(row["id"], alert_id)
                self.assertEqual(row["event_type"], "Crowd Density")
                self.assertEqual(row["severity"], "Medium")
                self.assertEqual(row["source"], "test source")
        finally:
            database.DB_PATH = old_path
            database._initialized_path = old_initialized_path


if __name__ == "__main__":
    unittest.main()
