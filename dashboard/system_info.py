"""Read-only runtime and model status shown in Settings."""

import sqlite3
from contextlib import closing

import streamlit as st

from config import (
    ALERT_COOLDOWN_SECONDS,
    ALTERCATION_CONFIRM_FRAMES,
    ALTERCATION_PROXIMITY_SCALE,
    CAMERA_SOURCE,
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


def application_status() -> str:
    if st.session_state.get("last_processing_error"):
        return "Error"
    if st.session_state.get("detection_running"):
        return "Processing"
    if st.session_state.get("last_summary") is not None:
        return "Completed"
    return "System Ready"


def current_source_label() -> str:
    source = st.session_state.get("selected_source")
    if not source:
        return "Not selected"
    if source == "Sample Video":
        return st.session_state.get("selected_sample_video_path") or "Sample video not selected"
    if source == "Upload Video":
        uploaded = st.session_state.get("selected_uploaded_video")
        return getattr(uploaded, "name", "Upload not selected")
    if source == "Webcam":
        return f"Local camera {CAMERA_SOURCE}"
    return str(source)


def database_status() -> tuple[str, str]:
    if not DB_PATH.is_file():
        return "Unavailable", "Database file does not exist yet."
    try:
        with closing(sqlite3.connect(DB_PATH, timeout=2)) as connection:
            connection.execute("SELECT 1 FROM alerts LIMIT 1").fetchone()
        return "Connected", str(DB_PATH)
    except sqlite3.Error as exc:
        return "Error", str(exc)


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

    db_state, db_detail = database_status()
    sample_count = len(list_sample_videos())
    cells = st.columns(4)
    cells[0].metric("Detection model", YOLO_MODEL_PATH.name, "Loaded" if model_loaded else "Available · lazy load")
    cells[1].metric("Database", db_state)
    cells[2].metric("Valid sample videos", sample_count)
    cells[3].metric("Application status", application_status())

    with st.container(border=True):
        st.markdown("#### Runtime")
        st.write(f"**Current source:** {current_source_label()}")
        st.write(f"**Model file:** `{YOLO_MODEL_PATH}` — {'present' if YOLO_MODEL_PATH.is_file() else 'missing'}")
        st.write(f"**Database:** `{db_detail}`")
        st.write(f"**Frame safety limit:** {MAX_PROCESSING_FRAMES}")
        st.write(f"**YOLO confidence:** {confidence:.2f}")

    st.markdown("#### Active detection modules")
    modules = [
        ("People / objects", "YOLOv8n · COCO model", "Trained object detector"),
        ("Fall assessment", f"Upright-to-horizontal persistence · {FALL_PERSISTENCE_SECONDS:.1f}s", "Heuristic"),
        ("Crowd assessment", f"{CROWD_COUNT_THRESHOLD}+ people · {CROWD_OCCUPANCY_THRESHOLD:.0%} occupancy", "Count + bounding-box occupancy"),
        ("High crowd severity", f"{CROWD_HIGH_COUNT_THRESHOLD}+ people · {CROWD_HIGH_OCCUPANCY_THRESHOLD:.0%} occupancy", "Configured threshold"),
        ("Possible altercation", f"Nearby scale {ALTERCATION_PROXIMITY_SCALE:.1f} · {ALTERCATION_CONFIRM_FRAMES} frames", "Experimental optical-flow heuristic; not a fight model"),
        ("Motion thresholds", f"Magnitude > {OPTICAL_FLOW_MAGNITUDE_THRESHOLD:.1f} · moving fraction {OPTICAL_FLOW_MOTION_FRACTION:.0%}", "Altercation heuristic settings"),
        ("Alert cooldown", f"{st.session_state.get('alert_cooldown_seconds', ALERT_COOLDOWN_SECONDS):.0f} seconds", "Per event and source, in memory"),
    ]
    for title, value, note in modules:
        with st.container(border=True):
            st.markdown(f"**{title}** · {value}")
            st.caption(note)
