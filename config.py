"""Central configuration values for the AntaDrishti prototype."""

from pathlib import Path

PROJECT_NAME = "AntaDrishti"
APP_NAME = "ANTAHDRISHTI"
APP_DESCRIPTION = "AI-Powered Public Safety Intelligence System"
CAMERA_SOURCE = 0

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "alerts.db"
OUTPUT_DIR = BASE_DIR / "outputs"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
SAMPLE_VIDEO_DIR = BASE_DIR / "sample_videos"
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"

ALERT_THRESHOLDS = {
    "fall": 1,
    "violence": 1,
    "crowd": 5,
    "bag": 1,
}

COLORS = {
    "danger": "#ff4b4b",
    "warning": "#ffbf47",
    "success": "#21c354",
    "info": "#4da3ff",
}

SEVERITY_LEVELS = {
    "Low": (0, 3),
    "Medium": (4, 6),
    "High": (7, 10),
}
