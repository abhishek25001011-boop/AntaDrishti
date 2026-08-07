"""Video utilities for the AntaDrishti detection pipeline."""

import cv2
import tempfile
from pathlib import Path
from typing import Optional

from config import SAMPLE_VIDEO_DIR


def list_sample_videos() -> list[str]:
    """Return available sample videos from the workspace."""
    if not SAMPLE_VIDEO_DIR.exists():
        return []

    return [str(path) for path in sorted(SAMPLE_VIDEO_DIR.glob("*")) if path.is_file()]


def open_video_capture(source: str, uploaded_file=None, sample_video_path: str | None = None) -> Optional[cv2.VideoCapture]:
    """Open a cv2.VideoCapture for webcam, uploaded video, or sample video."""
    try:
        if source == "Webcam":
            print("Opening video source: webcam")
            return cv2.VideoCapture(0)

        if source in {"Uploaded Video", "Upload Video"}:
            if uploaded_file is None:
                return None

            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_file.name).suffix,
                delete=False,
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            print(f"Opening video source: {temp_path}")
            capture = cv2.VideoCapture(temp_path)
            if capture.isOpened():
                return capture
            print(f"Opening video source: {temp_path} with CAP_FFMPEG")
            return cv2.VideoCapture(temp_path, cv2.CAP_FFMPEG)

        if source == "Sample Video":
            preferred_names = ["demo_valid.mp4", "crowd.mp4", "fall.mp4", "fight.mp4", "bag.mp4"]
            candidate_paths: list[Path] = []

            if sample_video_path:
                selected_path = Path(sample_video_path)
                if selected_path.exists():
                    candidate_paths.append(selected_path)

            for file_name in preferred_names:
                candidate_path = SAMPLE_VIDEO_DIR / file_name
                if candidate_path.exists() and candidate_path not in candidate_paths:
                    candidate_paths.append(candidate_path)

            for candidate_path in candidate_paths:
                resolved_path = candidate_path.resolve()
                print(f"Opening video source: {resolved_path}")
                capture = cv2.VideoCapture(str(resolved_path))
                if capture.isOpened():
                    return capture

                print(f"Opening video source: {resolved_path} with CAP_FFMPEG")
                capture = cv2.VideoCapture(str(resolved_path), cv2.CAP_FFMPEG)
                if capture.isOpened():
                    return capture

                print(f"Opening video source: {resolved_path} with CAP_ANY")
                capture = cv2.VideoCapture(str(resolved_path), cv2.CAP_ANY)
                if capture.isOpened():
                    return capture

            return None

        return None
    except Exception:
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
