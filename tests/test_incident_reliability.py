"""Focused regressions for event state and incident persistence."""

import json
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import database.database as database
from detectors.detection_manager import DetectionManager, DetectionResult
from detectors.person_detector import get_person_detections
from utils.alert_cooldown import AlertCooldown
from utils.event_detection import DetectedEvent, EventDetector
from utils.video_pipeline import process_capture
from utils.video_pipeline import PipelineSummary


class IncidentReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.old_db_path = database.DB_PATH
        self.old_initialized_path = database._initialized_path

    def tearDown(self):
        database.DB_PATH = self.old_db_path
        database._initialized_path = self.old_initialized_path

    def _use_temp_database(self, folder):
        database.DB_PATH = Path(folder) / "test.db"
        database._initialized_path = None

    def test_malformed_or_missing_detections_are_ignored(self):
        detections = [
            None,
            {},
            {"label": "person"},
            {"label": "person", "box": (0, 0, float("nan"), 10)},
            {"label": "person", "box": (10, 10, 0, 0)},
            {"label": "suitcase", "box": None},
        ]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        self.assertEqual(get_person_detections(None), [])
        self.assertEqual(get_person_detections(detections), [])
        self.assertEqual(EventDetector().process_frame(frame, None, now=0), [])
        self.assertEqual(EventDetector().process_frame(frame, detections, now=0), [])

    def test_model_is_loaded_once_across_manager_reruns(self):
        manager = DetectionManager()
        previous_model = manager._model
        manager._model = None
        model = object()
        try:
            with patch("detectors.detection_manager.load_yolo_model", return_value=model) as load:
                self.assertIs(manager.get_model(), model)
                self.assertIs(DetectionManager().get_model(), model)
                self.assertEqual(load.call_count, 1)
        finally:
            manager._model = previous_model

    def test_crowd_event_starts_once_and_restarts_after_clear(self):
        detector = EventDetector()
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        group = [
            {"label": "person", "box": (i * 75, 0, i * 75 + 70, 70)}
            for i in range(5)
        ]

        first = detector.process_frame(frame, group, now=0)
        continuation = detector.process_frame(frame, group, now=1)
        detector.process_frame(frame, [], now=2)
        detector.process_frame(frame, [], now=6)
        after_clear = detector.process_frame(frame, group, now=7)

        self.assertEqual([event.event_type for event in first], ["Crowd Density"])
        self.assertEqual(continuation, [])
        self.assertEqual([event.event_type for event in after_clear], ["Crowd Density"])

    def test_snapshot_failure_still_persists_incident_without_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            self._use_temp_database(folder)
            frame = np.zeros((20, 30, 3), dtype=np.uint8)
            with patch("utils.snapshot.save_snapshot", side_effect=OSError("disk full")):
                from utils.alert_engine import record_incident

                incident = record_incident(
                    frame,
                    "Possible Fall (posture heuristic)",
                    "High",
                    "review",
                    "cam",
                    metadata={"box": [2, 3, 10, 18]},
                )

            row = dict(database.fetch_alerts(1)[0])
            self.assertEqual(incident["snapshot_path"], None)
            self.assertEqual(row["event_type"], "Possible Fall (posture heuristic)")
            self.assertEqual(row["severity"], "High")
            self.assertEqual(row["source"], "cam")
            self.assertTrue(row["timestamp"])
            self.assertIsNone(row["snapshot_path"])
            self.assertEqual(json.loads(row["metadata"]), {"box": [2, 3, 10, 18]})

    def test_incident_recording_saves_snapshot_and_metadata_to_sqlite(self):
        with tempfile.TemporaryDirectory() as folder:
            self._use_temp_database(folder)
            frame = np.zeros((20, 30, 3), dtype=np.uint8)
            with patch("config.SNAPSHOT_DIR", Path(folder) / "snapshots"):
                from utils.alert_engine import record_incident

                incident = record_incident(
                    frame,
                    "Bag Detection",
                    "Low",
                    "backpack detected",
                    "sample.mp4",
                    metadata={"box": [1, 2, 15, 19]},
                )

            row = dict(database.fetch_alerts(1)[0])
            self.assertEqual(row["id"], incident["id"])
            self.assertEqual(row["event_type"], "Bag Detection")
            self.assertEqual(row["severity"], "Low")
            self.assertEqual(row["source"], "sample.mp4")
            self.assertEqual(json.loads(row["metadata"]), {"box": [1, 2, 15, 19]})
            self.assertTrue(Path(row["snapshot_path"]).is_file())

    def test_video_event_reaches_real_snapshot_sqlite_and_history_reader(self):
        from dashboard.history import incidents_frame

        class Capture:
            def __init__(self):
                self.reads = 0

            def get(self, prop):
                return 1

            def read(self):
                self.reads += 1
                return (self.reads == 1, np.zeros((40, 50, 3), dtype=np.uint8))

        class Manager:
            def process_frame(self, frame, confidence):
                detections = [{
                    "label": "backpack", "confidence": 0.92,
                    "box": (4, 5, 20, 30),
                }]
                return DetectionResult(frame.copy(), detections, 0, 1, 10.0, "LOW", 0.0)

        with tempfile.TemporaryDirectory() as folder:
            self._use_temp_database(folder)
            with patch("config.SNAPSHOT_DIR", Path(folder) / "snapshots"):
                summary = process_capture(
                    Capture(), "integration sample", Manager(), EventDetector(),
                    AlertCooldown(), frame_limit=1,
                )

            self.assertEqual(len(summary.incidents), 1)
            incident = summary.incidents[0]
            row = dict(database.fetch_alerts(limit=1)[0])
            self.assertEqual(row["id"], incident["id"])
            self.assertEqual(row["event_type"], "Bag Detection")
            self.assertEqual(row["severity"], "Low")
            self.assertEqual(row["source"], "integration sample")
            self.assertTrue(row["timestamp"])
            self.assertEqual(json.loads(row["metadata"]), {"box": [4, 5, 20, 30]})
            snapshot = Path(row["snapshot_path"])
            self.assertTrue(snapshot.is_file())
            history = incidents_frame()
            self.assertEqual(history.iloc[0]["id"], row["id"])
            self.assertEqual(history.iloc[0]["snapshot_path"], str(snapshot))

    def test_legacy_database_migrates_metadata_column_without_losing_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            db_path = Path(folder) / "legacy.db"
            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.execute(
                    """CREATE TABLE alerts (
                        id INTEGER PRIMARY KEY, event_type TEXT, severity TEXT,
                        message TEXT, source TEXT, snapshot_path TEXT,
                        timestamp TEXT, status TEXT, alert_type TEXT, created_at TEXT
                    )"""
                )
                conn.execute(
                    "INSERT INTO alerts (id, event_type, severity, message, source, timestamp, status) "
                    "VALUES (1, 'Crowd Density', 'Medium', 'legacy', 'cam', '2026-01-01', 'Active')"
                )
            database.DB_PATH = db_path
            database._initialized_path = None

            row = dict(database.fetch_alerts(1)[0])
            with closing(sqlite3.connect(db_path)) as conn:
                columns = {item[1] for item in conn.execute("PRAGMA table_info(alerts)")}

            self.assertIn("metadata", columns)
            self.assertEqual(row["event_type"], "Crowd Density")
            self.assertIsNone(row["metadata"])
            webcam_rows = database.fetch_recent_alerts_by_source("cam", limit=10)
            self.assertEqual([item["id"] for item in webcam_rows], [1])

    def test_live_timeline_formats_saved_timestamp_in_local_time(self):
        from dashboard.live_feed import _incident_description, format_saved_timestamp_local

        saved_utc = "2026-04-05 06:07:08"
        expected = datetime(
            2026, 4, 5, 6, 7, 8, tzinfo=timezone.utc
        ).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

        self.assertEqual(format_saved_timestamp_local(saved_utc), expected)
        self.assertEqual(format_saved_timestamp_local(None), "Unavailable")
        self.assertEqual(format_saved_timestamp_local("not a timestamp"), "Unavailable")
        self.assertEqual(
            format_saved_timestamp_local("2026-04-05T06:07:08.123456+00:00"),
            datetime.fromisoformat("2026-04-05T06:07:08.123456+00:00")
            .astimezone().strftime("%Y-%m-%d %H:%M:%S.%f %z"),
        )
        self.assertEqual(_incident_description({"message": "Possible event; review required."}), "Possible event; review required.")
        self.assertEqual(_incident_description({"message": "", "metadata": None}), "Unavailable")

    def test_recent_live_incidents_are_bounded_filtered_and_newest_first(self):
        with tempfile.TemporaryDirectory() as folder:
            self._use_temp_database(folder)
            older_id = database.save_alert(
                "Crowd Density", "Medium", "older", source="Browser webcam",
                timestamp="2026-01-01 12:00:00",
            )
            newest_id = database.save_alert(
                "Bag Detection", "Low", "newest", source="Browser webcam",
                timestamp="2026-01-03 12:00:00",
            )
            database.save_alert(
                "Bag Detection", "Low", "uploaded video", source="bag.mp4",
                timestamp="2026-01-04 12:00:00",
            )

            rows = database.fetch_recent_alerts_by_source("Browser webcam", limit=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], newest_id)
        self.assertNotEqual(rows[0]["id"], older_id)
        self.assertEqual(rows[0]["source"], "Browser webcam")

    def test_live_timeline_partitions_and_deduplicates_persisted_rows(self):
        from dashboard.live_feed import _partition_live_incidents

        rows = [
            {"id": 7, "event_type": "Crowd Density", "timestamp": "2026-01-01 10:00:00"},
            {"id": 8, "event_type": "Bag Detection", "timestamp": "2026-01-02 10:00:00"},
            {"id": 8, "event_type": "Bag Detection", "timestamp": "2026-01-02 10:00:00"},
        ]
        current, previous = _partition_live_incidents(rows, [8])

        self.assertEqual([row["id"] for row in current], [8])
        self.assertEqual([row["id"] for row in previous], [7])

    def test_live_snapshot_gallery_handles_missing_unreadable_and_outside_paths(self):
        import cv2
        from dashboard import live_feed

        with tempfile.TemporaryDirectory() as folder:
            snapshot_dir = Path(folder) / "snapshots"
            snapshot_dir.mkdir()
            valid_path = snapshot_dir / "valid.jpg"
            unreadable_path = snapshot_dir / "unreadable.jpg"
            outside_path = Path(folder) / "outside.jpg"
            cv2.imwrite(str(valid_path), np.zeros((12, 16, 3), dtype=np.uint8))
            unreadable_path.write_text("not an image", encoding="utf-8")
            outside_path.write_bytes(valid_path.read_bytes())

            with patch.object(live_feed, "SNAPSHOT_DIR", snapshot_dir):
                preview, error = live_feed._load_incident_snapshot(str(valid_path))
                missing, missing_error = live_feed._load_incident_snapshot(str(snapshot_dir / "gone.jpg"))
                unreadable, unreadable_error = live_feed._load_incident_snapshot(str(unreadable_path))
                outside, outside_error = live_feed._load_incident_snapshot(str(outside_path))
                absent, absent_error = live_feed._load_incident_snapshot(None)

        self.assertEqual(preview.shape, (12, 16, 3))
        self.assertIsNone(error)
        self.assertIsNone(missing)
        self.assertIn("missing", missing_error)
        self.assertIsNone(unreadable)
        self.assertIn("unreadable", unreadable_error)
        self.assertIsNone(outside)
        self.assertEqual(outside_error, "Snapshot unavailable.")
        self.assertIsNone(absent)
        self.assertEqual(absent_error, "No snapshot was saved.")

    def test_live_timeline_shows_empty_state_without_saved_incidents(self):
        from dashboard import live_feed

        class StreamlitStub:
            def __init__(self):
                self.session_state = {"webcam_session_incident_ids": []}
                self.messages = []

            def markdown(self, message):
                self.messages.append(message)

            def selectbox(self, *args, **kwargs):
                return 20

            def info(self, message):
                self.messages.append(message)

            def warning(self, message):
                self.messages.append(message)

        stub = StreamlitStub()
        with tempfile.TemporaryDirectory() as folder:
            self._use_temp_database(folder)
            with patch.object(live_feed, "st", stub):
                live_feed._render_live_incident_timeline()

        self.assertTrue(any("No incidents have been persisted" in item for item in stub.messages))
        self.assertTrue(any("Recorded Evidence (Snapshots)" in item for item in stub.messages))

    def test_database_failure_removes_created_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            self._use_temp_database(folder)
            frame = np.zeros((20, 30, 3), dtype=np.uint8)
            with patch("config.SNAPSHOT_DIR", Path(folder) / "snapshots"):
                with patch("database.database.save_alert", side_effect=RuntimeError("db unavailable")):
                    from utils.alert_engine import record_incident

                    with self.assertRaisesRegex(RuntimeError, "db unavailable"):
                        record_incident(frame, "Crowd Density", "Medium", "review", "cam")
            self.assertEqual(list((Path(folder) / "snapshots").glob("*.jpg")), [])

    def test_video_pipeline_suppresses_repeated_event_and_releases_cooldown_on_failure(self):
        class Capture:
            def __init__(self, count):
                self.count = count
                self.reads = 0

            def get(self, prop):
                return self.count

            def read(self):
                self.reads += 1
                return (self.reads <= self.count, np.zeros((24, 24, 3), dtype=np.uint8))

        class Manager:
            def process_frame(self, frame, confidence):
                return DetectionResult(frame.copy(), [], 0, 0, 10.0, "LOW", 0.0)

        event = DetectedEvent("Bag Detection", "Low", "test event", (1, 2, 8, 9))
        detector = SimpleNamespace(process_frame=lambda frame, detections: [event])
        cooldown = AlertCooldown(30)
        capture = Capture(3)
        with patch("utils.video_pipeline.record_incident", return_value={"id": 1}) as persist:
            summary = process_capture(capture, "cam", Manager(), detector, cooldown, frame_limit=3)
        self.assertEqual(summary.frames_processed, 3)
        self.assertEqual(persist.call_count, 1)

        retry_cooldown = AlertCooldown(30)
        with patch(
            "utils.video_pipeline.record_incident",
            side_effect=[RuntimeError("db unavailable"), {"id": 2}],
        ) as persist:
            with self.assertRaisesRegex(RuntimeError, "db unavailable"):
                process_capture(Capture(1), "retry", Manager(), detector, retry_cooldown, frame_limit=1)
            summary = process_capture(Capture(1), "retry", Manager(), detector, retry_cooldown, frame_limit=1)
        self.assertEqual(summary.incidents, [{"id": 2}])
        self.assertEqual(persist.call_count, 2)

    def test_video_pipeline_zero_event_summary_keeps_frame_and_people_counts(self):
        class Capture:
            reads = 0

            def get(self, prop):
                return 2

            def read(self):
                self.reads += 1
                return (self.reads <= 2, np.zeros((12, 12, 3), dtype=np.uint8))

        class Manager:
            def process_frame(self, frame, confidence):
                return DetectionResult(frame.copy(), [], 3, 0, 10.0, "LOW", 0.0)

        summary = process_capture(
            Capture(), "zero-event video", Manager(), EventDetector(),
            AlertCooldown(), frame_limit=3,
        )

        self.assertEqual(summary.frames_processed, 2)
        self.assertTrue(summary.reached_end)
        self.assertEqual(summary.incidents, [])
        self.assertEqual(summary.peak_people_count, 3)

    def test_video_results_render_zero_events_and_incidents_after_rerun(self):
        from dashboard import live_feed

        class StreamlitStub:
            def __init__(self, state):
                self.session_state = state
                self.messages = []
                self.tables = []
                self.images = []

            def empty(self):
                return self

            def columns(self, count):
                return [self for _ in range(count)]

            def container(self, **kwargs):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def metric(self, label, value, *args, **kwargs):
                self.messages.append(f"{label}: {value}")
                return self

            def subheader(self, text):
                self.messages.append(text)

            def markdown(self, text, **kwargs):
                self.messages.append(text)

            def caption(self, text):
                self.messages.append(text)

            def success(self, text):
                self.messages.append(text)

            def info(self, text):
                self.messages.append(text)

            def error(self, text):
                self.messages.append(text)

            def warning(self, text):
                self.messages.append(text)

            def dataframe(self, rows, **kwargs):
                self.tables.append(rows)

            def image(self, path, **kwargs):
                self.images.append(path)

        class SessionState(dict):
            __getattr__ = dict.__getitem__

            def __setattr__(self, name, value):
                self[name] = value

        state = SessionState({
            "detection_running": False,
            "last_summary": PipelineSummary(frames_processed=4, reached_end=True),
            "last_run_source": "quiet.mp4",
            "last_processing_error": None,
        })
        stub = StreamlitStub(state)
        with patch.object(live_feed, "st", stub):
            live_feed.render_live_detection()
        self.assertIn("Analysis completed - no incidents detected", stub.messages)
        self.assertIn("Frames processed: 4", " ".join(stub.messages))

        with tempfile.TemporaryDirectory() as folder:
            self._use_temp_database(folder)
            snapshot = Path(folder) / "incident.jpg"
            snapshot.write_bytes(b"test snapshot placeholder")
            database.save_alert(
                "Crowd Density", "Medium", "webcam event", source="Browser webcam",
            )
            alert_id = database.save_alert(
                "Bag Detection", "Low", "backpack detected", source="event.mp4",
                snapshot_path=str(snapshot), timestamp="2026-01-02 03:04:05",
            )
            state["last_summary"] = PipelineSummary(
                frames_processed=5,
                reached_end=True,
                incidents=[{"id": alert_id}],
            )
            state["last_run_source"] = "event.mp4"
            stub = StreamlitStub(state)
            with patch.object(live_feed, "st", stub):
                live_feed.render_live_detection()

        self.assertIn("Analysis completed - 1 incident(s) detected and saved.", stub.messages)
        self.assertEqual(stub.tables[0][0]["Event type"], "Bag Detection")
        self.assertEqual(stub.tables[0][0]["Severity"], "Low")
        self.assertEqual(stub.tables[0][0]["Local time"], live_feed.format_saved_timestamp_local("2026-01-02 03:04:05"))
        self.assertEqual(len(stub.tables[0]), 1)
        self.assertEqual(stub.tables[0][0]["Snapshot"], "Unavailable")
        self.assertEqual(stub.images, [])

    def test_video_analysis_empty_state_explains_webcam_navigation_without_mixing_results(self):
        from dashboard import live_feed

        class StreamlitStub:
            def __init__(self, state):
                self.session_state = state
                self.messages = []

            def empty(self):
                return self

            def columns(self, count):
                return [self for _ in range(count)]

            def metric(self, label, value, *args, **kwargs):
                self.messages.append(f"{label}: {value}")
                return self

            def info(self, message):
                self.messages.append(message)

            def caption(self, message):
                self.messages.append(message)

        class SessionState(dict):
            __getattr__ = dict.__getitem__

            def __setattr__(self, name, value):
                self[name] = value

        webcam_event = DetectedEvent("Crowd Density", "Medium", "webcam-only event")
        stub = StreamlitStub(SessionState({
            "detection_running": False,
            "last_summary": None,
            "last_processing_error": None,
            "webcam_incidents": [webcam_event],
        }))
        with patch.object(live_feed, "st", stub):
            live_feed.render_live_detection()

        rendered = " ".join(stub.messages)
        self.assertIn("Choose a source in the sidebar", rendered)
        self.assertIn("browser webcam results are shown under Live Detection", rendered)
        self.assertIn("Persisted incidents from either workflow", rendered)
        self.assertNotIn("Analysis results", rendered)
        self.assertNotIn("webcam-only event", rendered)

    def test_video_analysis_error_remains_visible_when_idle_after_rerun(self):
        from dashboard import live_feed

        class StreamlitStub:
            def __init__(self, state):
                self.session_state = state
                self.messages = []

            def empty(self):
                return self

            def columns(self, count):
                return [self for _ in range(count)]

            def metric(self, label, value, *args, **kwargs):
                self.messages.append(f"{label}: {value}")
                return self

            def error(self, message):
                self.messages.append(message)

            def info(self, message):
                self.messages.append(message)

        class SessionState(dict):
            __getattr__ = dict.__getitem__

            def __setattr__(self, name, value):
                self[name] = value

        stub = StreamlitStub(SessionState({
            "detection_running": False,
            "last_summary": None,
            "last_processing_error": "Could not open this video. Check that the selected file is valid and nonempty.",
        }))
        with patch.object(live_feed, "st", stub):
            live_feed.render_live_detection()

        self.assertTrue(any("Could not open this video" in message for message in stub.messages))
        self.assertFalse(any("Choose a source" in message for message in stub.messages))

    def test_invalid_upload_error_is_retained_after_processing_rerun(self):
        from dashboard import live_feed

        class StreamlitStub:
            def __init__(self, state):
                self.session_state = state
                self.messages = []

            def empty(self):
                return self

            def columns(self, count):
                return [self for _ in range(count)]

            def metric(self, label, value, *args, **kwargs):
                self.messages.append(f"{label}: {value}")
                return self

            def error(self, message):
                self.messages.append(message)

        class SessionState(dict):
            __getattr__ = dict.__getitem__

            def __setattr__(self, name, value):
                self[name] = value

        state = SessionState({
            "detection_running": True,
            "selected_source": "Upload Video",
            "selected_uploaded_video": SimpleNamespace(name="broken.mp4"),
            "selected_sample_video_path": None,
            "detector_manager": None,
            "alert_cooldown": None,
        })
        stub = StreamlitStub(state)
        with patch.object(live_feed, "st", stub):
            with patch.object(live_feed, "open_video_capture", return_value=None):
                live_feed.render_live_detection()
            self.assertFalse(state["detection_running"])
            self.assertIn("valid and nonempty", state["last_processing_error"])
            live_feed.render_live_detection()

        self.assertGreaterEqual(len(stub.messages), 2)
        self.assertTrue(all("valid and nonempty" in message for message in stub.messages))

    def test_video_processing_exception_is_sanitized_and_retained_after_rerun(self):
        from dashboard import live_feed

        class Capture:
            def isOpened(self):
                return True

            def release(self):
                pass

        class Progress:
            def progress(self, *args, **kwargs):
                pass

        class StreamlitStub:
            def __init__(self, state):
                self.session_state = state
                self.messages = []

            def empty(self):
                return self

            def columns(self, count):
                return [self for _ in range(count)]

            def metric(self, label, value, *args, **kwargs):
                self.messages.append(f"{label}: {value}")
                return self

            def error(self, message):
                self.messages.append(message)

            def progress(self, *args, **kwargs):
                return Progress()

        class SessionState(dict):
            __getattr__ = dict.__getitem__

            def __setattr__(self, name, value):
                self[name] = value

        state = SessionState({
            "detection_running": True,
            "selected_source": "Sample Video",
            "selected_sample_video_path": "sample.mp4",
            "selected_uploaded_video": None,
            "detector_manager": object(),
            "alert_cooldown": AlertCooldown(),
        })
        stub = StreamlitStub(state)
        with patch.object(live_feed, "st", stub):
            with patch.object(live_feed, "open_video_capture", return_value=Capture()):
                with patch.object(live_feed, "process_capture", side_effect=RuntimeError("private path")):
                    live_feed.render_live_detection()
            self.assertEqual(state["last_processing_error"], "Video analysis failed.")
            self.assertNotIn("private path", " ".join(stub.messages))
            live_feed.render_live_detection()

        self.assertIn("Video analysis failed.", stub.messages)

    def test_webcam_persistence_failure_is_retained_and_retried_once(self):
        from dashboard import live_feed
        from utils.webcam_pipeline import WebcamUpdate

        class Processor:
            def __init__(self, updates=()):
                self.updates = list(updates)

            def drain(self):
                updates, self.updates = self.updates, []
                return updates

        class StreamlitStub:
            def __init__(self, state):
                self.session_state = state
                self.warnings = []

            def warning(self, message):
                self.warnings.append(message)

        class SessionState(dict):
            def __getattr__(self, name):
                return self[name]

            def __setattr__(self, name, value):
                self[name] = value

        event = DetectedEvent("Bag Detection", "Low", "test event", (1, 2, 8, 9))
        result = DetectionResult(np.zeros((10, 10, 3), dtype=np.uint8), [], 0, 0, 10.0, "LOW", 0.0)
        update = WebcamUpdate(result, [event])
        cooldown = AlertCooldown(30)
        state = SessionState({
            "alert_cooldown": cooldown,
            "webcam_pending_incidents": [],
            "webcam_incidents": [],
            "webcam_last_update": None,
            "webcam_persist_error": None,
            "webcam_session_started_at": None,
        })
        streamlit_stub = StreamlitStub(state)

        with patch.object(live_feed, "st", streamlit_stub):
            with patch(
                "utils.alert_engine.record_incident",
                side_effect=[RuntimeError("db unavailable"), {"id": 1}],
            ) as persist:
                live_feed._persist_webcam_updates(Processor([update]))
                self.assertEqual(len(state["webcam_pending_incidents"]), 1)
                self.assertEqual(
                    state["webcam_persist_error"], "Incident save failed; retrying."
                )

                live_feed._persist_webcam_updates(Processor())
                self.assertEqual(state["webcam_pending_incidents"], [])
                self.assertIsNone(state["webcam_persist_error"])
                self.assertEqual(state["webcam_incidents"], [event])
                self.assertEqual(state["webcam_session_incident_ids"], [1])

                live_feed._persist_webcam_updates(Processor([update]))
                self.assertEqual(persist.call_count, 2)
                self.assertEqual(persist.call_args.kwargs["metadata"], {"box": [1, 2, 8, 9]})

        self.assertEqual(len(streamlit_stub.warnings), 1)
        self.assertNotIn("db unavailable", streamlit_stub.warnings[0])

    def test_camera_session_reset_keeps_failed_incident_pending(self):
        from dashboard import live_feed

        class Processor:
            reset_called = False

            def drain(self):
                return []

            def reset(self):
                self.reset_called = True

        class StreamlitStub:
            def __init__(self, state):
                self.session_state = state
                self.warnings = []

            def warning(self, message):
                self.warnings.append(message)

        class SessionState(dict):
            def __getattr__(self, name):
                return self[name]

            def __setattr__(self, name, value):
                self[name] = value

        event = DetectedEvent("Bag Detection", "Low", "review", (1, 2, 8, 9))
        pending = [(np.zeros((8, 8, 3), dtype=np.uint8), event)]
        state = SessionState({
            "alert_cooldown": AlertCooldown(30),
            "webcam_pending_incidents": pending,
            "webcam_incidents": [event],
            "webcam_last_update": object(),
            "webcam_persist_error": "Incident save failed; retrying.",
            "last_detection_metrics": {
                "source": "Browser webcam", "updated_at": 0,
            },
            "webcam_session_started_at": None,
        })
        streamlit_stub = StreamlitStub(state)
        processor = Processor()

        with patch.object(live_feed, "st", streamlit_stub):
            with patch("utils.alert_engine.record_incident", side_effect=RuntimeError("db unavailable")):
                live_feed._reset_webcam_session(processor)

        self.assertTrue(processor.reset_called)
        self.assertIs(state["webcam_pending_incidents"], pending)
        self.assertEqual(state["webcam_persist_error"], "Incident save failed; retrying.")
        self.assertEqual(state["webcam_incidents"], [])
        self.assertEqual(state["webcam_session_incident_ids"], [])
        self.assertIsNone(state["last_detection_metrics"])

    def test_webcam_frame_handoff_reaches_sqlite_snapshot_and_history(self):
        from dashboard import live_feed
        from dashboard.history import incidents_frame
        from utils.webcam_pipeline import WebcamProcessor

        class VideoFrame:
            def __init__(self, frame):
                self.frame = frame

            def to_ndarray(self, format):
                return self.frame

            @classmethod
            def from_ndarray(cls, frame, format):
                return cls(frame)

        class Manager:
            def process_frame(self, frame, confidence):
                detections = [{
                    "label": "backpack", "confidence": 0.9,
                    "box": (2, 3, 18, 24),
                }]
                return DetectionResult(frame.copy(), detections, 0, 1, 10.0, "LOW", 0.0)

        class StreamlitStub:
            def __init__(self, state):
                self.session_state = state

            def warning(self, message):
                raise AssertionError(f"Unexpected persistence warning: {message}")

        frame = np.zeros((32, 40, 3), dtype=np.uint8)
        processor = WebcamProcessor(Manager(), process_every_n_frames=1)
        processor.callback(VideoFrame(frame))

        class SessionState(dict):
            __getattr__ = dict.__getitem__

            def __setattr__(self, name, value):
                self[name] = value

        state = SessionState({
            "alert_cooldown": AlertCooldown(30),
            "webcam_pending_incidents": [],
            "webcam_incidents": [],
            "webcam_last_update": None,
            "webcam_persist_error": None,
            # Simulate an old queued update arriving after a restart.
            "webcam_session_started_at": time.monotonic() + 1,
            "last_detection_metrics": None,
        })

        with tempfile.TemporaryDirectory() as folder:
            self._use_temp_database(folder)
            with patch("config.SNAPSHOT_DIR", Path(folder) / "snapshots"):
                with patch.object(live_feed, "st", StreamlitStub(state)):
                    live_feed._persist_webcam_updates(processor)

                row = dict(database.fetch_alerts(limit=1)[0])
                self.assertEqual(row["event_type"], "Bag Detection")
                self.assertEqual(row["source"], "Browser webcam")
                self.assertEqual(row["severity"], "Low")
                self.assertEqual(json.loads(row["metadata"]), {"box": [2, 3, 18, 24]})
                self.assertTrue(Path(row["snapshot_path"]).is_file())
                self.assertEqual(state["webcam_session_incident_ids"], [row["id"]])
                history = incidents_frame()
                self.assertEqual(history.iloc[0]["event_type"], "Bag Detection")
                self.assertEqual(history.iloc[0]["source"], "Browser webcam")

                # The alert center and analytics read this shared persisted history;
                # none filters out the Browser webcam source.
                alerts = database.fetch_alerts(limit=None)
                self.assertEqual(dict(alerts[0])["source"], "Browser webcam")
                self.assertEqual(history.iloc[0]["source"], "Browser webcam")
                class FilterColumn:
                    def selectbox(self, label, options, **kwargs):
                        return options[0]

                    def text_input(self, *args, **kwargs):
                        return ""

                    def button(self, *args, **kwargs):
                        return False

                class HistoryStreamlitStub:
                    def columns(self, *args, **kwargs):
                        return [FilterColumn() for _ in range(5)]

                from dashboard import history as history_dashboard
                with patch.object(history_dashboard, "st", HistoryStreamlitStub()):
                    visible_alerts = history_dashboard._filtered_history(history)
                self.assertEqual(visible_alerts.iloc[0]["source"], "Browser webcam")

                from dashboard import analytics as analytics_dashboard

                class AnalyticsColumn:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def metric(self, *args, **kwargs):
                        pass

                class AnalyticsStreamlitStub:
                    def __init__(self):
                        self.charts = []

                    def subheader(self, *args, **kwargs):
                        pass

                    def columns(self, count, *args, **kwargs):
                        return [AnalyticsColumn() for _ in range(count)]

                    def markdown(self, *args, **kwargs):
                        pass

                    def bar_chart(self, data, **kwargs):
                        self.charts.append(data)

                    def line_chart(self, data, **kwargs):
                        self.charts.append(data)

                    def caption(self, *args, **kwargs):
                        pass

                analytics_stub = AnalyticsStreamlitStub()
                with patch.object(analytics_dashboard, "st", analytics_stub):
                    analytics_dashboard.render_analytics_page()
                self.assertIn("Bag Detection", analytics_stub.charts[0].index)
                self.assertIsNone(state["webcam_last_update"])
                self.assertIsNone(state["last_detection_metrics"])

    def test_upload_capture_release_removes_temporary_file(self):
        sample = Path("sample_videos/demo_valid.mp4")
        if not sample.is_file():
            self.skipTest("bundled sample video is unavailable")
        from types import SimpleNamespace
        from utils.video_utils import _TEMP_CAPTURE_FILES, open_video_capture, release_capture

        upload = SimpleNamespace(name=sample.name, getvalue=sample.read_bytes)
        capture = open_video_capture("Upload Video", uploaded_file=upload)
        self.assertIsNotNone(capture)
        temporary_path = _TEMP_CAPTURE_FILES[id(capture)]
        self.assertTrue(temporary_path.exists())
        release_capture(capture)
        self.assertFalse(temporary_path.exists())
        self.assertNotIn(id(capture), _TEMP_CAPTURE_FILES)

    def test_upload_rejects_unsupported_extension_and_oversized_file(self):
        from utils import video_utils

        unsupported = SimpleNamespace(name="not-a-video.txt", getvalue=lambda: b"data")
        self.assertIsNone(video_utils.open_video_capture("Upload Video", uploaded_file=unsupported))

        oversized = SimpleNamespace(name="large.mp4", size=11, getvalue=lambda: self.fail("read oversized upload"))
        with patch("utils.video_utils.MAX_VIDEO_UPLOAD_MB", 0):
            self.assertIsNone(video_utils.open_video_capture("Upload Video", uploaded_file=oversized))

    def test_upload_temp_file_is_removed_if_video_open_raises(self):
        from utils.video_utils import open_video_capture

        created_paths = []
        upload = SimpleNamespace(name="clip.mp4", getvalue=lambda: b"not a valid video")

        def fail_open(path):
            created_paths.append(path)
            raise RuntimeError("decoder failed")

        with patch("utils.video_utils._open_file_capture", side_effect=fail_open):
            self.assertIsNone(open_video_capture("Upload Video", uploaded_file=upload))
        self.assertEqual(len(created_paths), 1)
        self.assertFalse(created_paths[0].exists())


if __name__ == "__main__":
    unittest.main()
