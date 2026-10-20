"""Read-only runtime and model status shown in Settings."""

import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import streamlit as st

from config import (
    ALERT_COOLDOWN_SECONDS,
    ALTERCATION_CONFIRM_FRAMES,
    ALTERCATION_PROXIMITY_SCALE,
    CROWD_COUNT_THRESHOLD,
    CROWD_HIGH_COUNT_THRESHOLD,
    CROWD_HIGH_OCCUPANCY_THRESHOLD,
    CROWD_OCCUPANCY_THRESHOLD,
    FALL_PERSISTENCE_SECONDS,
    MAX_PROCESSING_FRAMES,
    OPTICAL_FLOW_MAGNITUDE_THRESHOLD,
    OPTICAL_FLOW_MOTION_FRACTION,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_MODEL_PATH,
    DB_PATH,
)
from utils.alert_cooldown import AlertCooldown
from utils.video_utils import list_sample_videos

logger = logging.getLogger(__name__)


def application_status() -> str:
    if st.session_state.get("webcam_stream_active"):
        metrics = st.session_state.get("last_detection_metrics") or {}
        if (
            metrics.get("source") == "Browser webcam"
            and time.time() - metrics.get("updated_at", 0) <= 5
        ):
            return "Detection Active"
        return "Camera Streaming"
    if st.session_state.get("detection_running"):
        return "Processing"
    if st.session_state.get("navigation") == "Video Analysis" and st.session_state.get("last_processing_error"):
        return "Error"
    if (
        st.session_state.get("navigation") == "Video Analysis"
        and st.session_state.get("last_summary") is not None
    ):
        return "Completed"
    return "Idle"


def current_source_label() -> str:
    if st.session_state.get("webcam_stream_active"):
        return "Browser webcam stream"
    if st.session_state.get("navigation") == "Live Detection":
        return "Browser webcam · idle"
    source = st.session_state.get("selected_source")
    if not source:
        return "Not selected"
    if source == "Sample Video":
        path = st.session_state.get("selected_sample_video_path")
        return Path(path).name if path else "Sample video not selected"
    if source == "Upload Video":
        uploaded = st.session_state.get("selected_uploaded_video")
        name = getattr(uploaded, "name", "Upload not selected")
        return Path(name).name
    if source == "Webcam":
        return "Local camera input selected"
    return str(source)


def database_status() -> tuple[str, str]:
    if not DB_PATH.is_file():
        return "Unavailable", "Database file does not exist yet."
    try:
        with closing(sqlite3.connect(DB_PATH, timeout=2)) as connection:
            connection.execute("SELECT 1 FROM alerts LIMIT 1").fetchone()
        return "Connected", "SQLite incident store is available."
    except sqlite3.Error:
        logger.exception("Could not query incident database status")
        return "Error", "Could not query the incident database."


def _apply_cooldown_setting() -> None:
    st.session_state.alert_cooldown = AlertCooldown(
        st.session_state.alert_cooldown_seconds
    )


def render_settings() -> None:
    st.subheader("Settings & system information")
    st.caption("Confidence is adjusted in source controls. Cooldown applies to this Streamlit session.")
    st.number_input(
        "Alert cooldown (seconds)", min_value=0.0, max_value=300.0,
        value=float(st.session_state.get("alert_cooldown_seconds", ALERT_COOLDOWN_SECONDS)),
        step=5.0, key="alert_cooldown_seconds", on_change=_apply_cooldown_setting,
        help="The same event from the same source will not save another alert or snapshot during this interval.",
    )
    confidence = st.session_state.get("conf_threshold", YOLO_CONFIDENCE_THRESHOLD)
    model_loaded = False
    manager = st.session_state.get("detector_manager")
    if manager is not None:
        model_loaded = getattr(manager, "_model", None) is not None
    model_available = YOLO_MODEL_PATH.is_file() and YOLO_MODEL_PATH.stat().st_size > 0

    db_state, db_detail = database_status()
    sample_count = len(list_sample_videos())
    cells = st.columns(4)
    model_state = "Loaded" if model_loaded else "Available · lazy load" if model_available else "Missing"
    cells[0].metric("Detection model", YOLO_MODEL_PATH.name, model_state)
    cells[1].metric("Database", db_state)
    cells[2].metric("Valid sample videos", sample_count)
    cells[3].metric("Application status", application_status())

    with st.container(border=True):
        st.markdown("#### Runtime")
        st.write(f"**Current source:** {current_source_label()}")
        st.write(f"**Model package:** `{YOLO_MODEL_PATH.name}` — {'present' if model_available else 'missing'}")
        st.write(f"**Database:** {db_detail}")
        camera_state = "Streaming" if st.session_state.get("webcam_stream_active") else "Permission not verified"
        st.write(f"**Browser camera:** {camera_state}")
        st.write(f"**Frame safety limit:** {MAX_PROCESSING_FRAMES}")
        st.write(f"**YOLO confidence:** {confidence:.2f}")

    st.markdown("#### Active detection modules")
    modules = [
        ("People / objects", "YOLOv8n / COCO model", "Trained object detector"),
        ("Fall assessment", f"Upright-to-horizontal persistence / {FALL_PERSISTENCE_SECONDS:.1f}s", "Heuristic"),
        ("Crowd assessment", f"{CROWD_COUNT_THRESHOLD}+ people / {CROWD_OCCUPANCY_THRESHOLD:.0%} occupancy", "Count + bounding-box occupancy"),
        ("High crowd severity", f"{CROWD_HIGH_COUNT_THRESHOLD}+ people / {CROWD_HIGH_OCCUPANCY_THRESHOLD:.0%} occupancy", "Configured threshold"),
        ("Possible altercation", f"Nearby scale {ALTERCATION_PROXIMITY_SCALE:.1f} / {ALTERCATION_CONFIRM_FRAMES} frames", "Experimental optical-flow heuristic; not a fight model"),
        ("Motion thresholds", f"Magnitude > {OPTICAL_FLOW_MAGNITUDE_THRESHOLD:.1f} / moving fraction {OPTICAL_FLOW_MOTION_FRACTION:.0%}", "Altercation heuristic settings"),
        ("Alert cooldown", f"{st.session_state.get('alert_cooldown_seconds', ALERT_COOLDOWN_SECONDS):.0f} seconds", "Per event and source, in memory"),
    ]
    st.dataframe(
        [{"Module": title, "Configuration": value, "Implementation": note} for title, value, note in modules],
        width="stretch",
        hide_index=True,
    )
