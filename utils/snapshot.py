"""Helpers for saving image snapshots from the camera feed.

TODO: Add timestamped file naming and image compression later.
"""

from pathlib import Path


def save_snapshot(frame, output_dir, prefix="snapshot"):
    """Save a frame to disk and return the output path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}.jpg"
    # TODO: Use Pillow or OpenCV to write the frame.
    return str(path)
