"""Helpers for saving image snapshots from the camera feed.

TODO: Add timestamped file naming and image compression later.
"""

from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

import cv2


def save_snapshot(frame, output_dir, prefix="snapshot"):
    """Save a frame to disk and return the output path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("Cannot save an empty frame")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_dir / f"{prefix}_{timestamp}_{uuid4().hex[:8]}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Could not write snapshot to {path}")
    return str(path)
