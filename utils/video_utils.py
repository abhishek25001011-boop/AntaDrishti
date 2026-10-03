"""Video utilities for the AntaDrishti detection pipeline."""

import cv2
import tempfile
from pathlib import Path
from typing import Optional

from config import CAMERA_SOURCE, SAMPLE_VIDEO_DIR

_TEMP_CAPTURE_FILES: dict[int, Path] = {}


def list_sample_videos() -> list[str]:
    """Return available sample videos from the workspace."""
    if not SAMPLE_VIDEO_DIR.exists():
        return []

    return [str(path) for path in sorted(SAMPLE_VIDEO_DIR.glob("*")) if path.is_file() and path.stat().st_size > 0]


def open_video_capture(source: str, uploaded_file=None, sample_video_path: str | None = None) -> Optional[cv2.VideoCapture]:
    """Open a cv2.VideoCapture for webcam, uploaded video, or sample video."""
    try:
        if source == "Webcam":
            print(f"Opening local camera source: {CAMERA_SOURCE}")
            return cv2.VideoCapture(CAMERA_SOURCE)

        if source in {"Uploaded Video", "Upload Video"}:
            if uploaded_file is None:
                return None

            with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            capture = _open_file_capture(Path(temp_path))
            if capture is None:
                Path(temp_path).unlink(missing_ok=True)
            else:
                _TEMP_CAPTURE_FILES[id(capture)] = Path(temp_path)
            return capture

        if source == "Sample Video":
            if not sample_video_path:
                return None
            selected_path = Path(sample_video_path)
            if not selected_path.is_file() or selected_path.stat().st_size == 0:
                return None
            return _open_file_capture(selected_path)

        return None
    except Exception:
        return None


def _open_file_capture(path: Path) -> Optional[cv2.VideoCapture]:
    """Open a non-empty video and reject files that cannot yield a frame."""
    resolved_path = str(path.resolve())
    for backend in (None, cv2.CAP_FFMPEG, cv2.CAP_ANY):
        capture = cv2.VideoCapture(resolved_path) if backend is None else cv2.VideoCapture(resolved_path, backend)
        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return capture
        capture.release()
    return None


def release_capture(capture: Optional[cv2.VideoCapture]) -> None:
    """Release a video capture handle if it is open."""
    if capture is None:
        return

    try:
        if capture.isOpened():
            capture.release()
    except Exception:
        pass
    temp_path = _TEMP_CAPTURE_FILES.pop(id(capture), None)
    if temp_path is not None:
        temp_path.unlink(missing_ok=True)
