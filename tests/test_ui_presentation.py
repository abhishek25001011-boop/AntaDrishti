from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from dashboard import history


class HistoryPresentationTests(unittest.TestCase):
    def test_timestamp_is_formatted_in_local_time(self):
        value = "2026-10-20T10:30:00+00:00"
        expected = datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        self.assertEqual(history._local_timestamp(value), expected)

    def test_missing_or_invalid_timestamp_is_unavailable(self):
        self.assertEqual(history._local_timestamp(""), "Unavailable")
        self.assertEqual(history._local_timestamp("not-a-time"), "Unavailable")

    def test_saved_media_sources_hide_local_directory_paths(self):
        self.assertEqual(history._source_label(r"C:\\private\\samples\\demo.mp4"), "demo.mp4")
        self.assertEqual(history._source_label("Browser webcam"), "Browser webcam")

    def test_snapshot_reader_accepts_valid_files_under_snapshot_root(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "evidence.jpg"
            self.assertTrue(cv2.imwrite(str(image_path), np.zeros((12, 16, 3), dtype=np.uint8)))
            with patch.object(history, "SNAPSHOT_DIR", Path(folder)):
                image = history._safe_snapshot(str(image_path))
            self.assertIsNotNone(image)
            self.assertEqual(image.shape, (12, 16, 3))

    def test_snapshot_reader_rejects_unreadable_and_outside_files(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            invalid_path = Path(folder) / "invalid.jpg"
            invalid_path.write_bytes(b"not an image")
            outside_path = Path(outside) / "outside.jpg"
            self.assertTrue(cv2.imwrite(str(outside_path), np.zeros((12, 16, 3), dtype=np.uint8)))
            with patch.object(history, "SNAPSHOT_DIR", Path(folder)):
                self.assertIsNone(history._safe_snapshot(str(invalid_path)))
                self.assertIsNone(history._safe_snapshot(str(outside_path)))


if __name__ == "__main__":
    unittest.main()
