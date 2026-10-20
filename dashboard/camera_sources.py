"""Honest status view for camera workflows supported by the current app."""

from __future__ import annotations

import time

import streamlit as st

from config import CAMERA_SOURCE
from dashboard.live_feed import format_saved_timestamp_local
from database.database import fetch_recent_alerts_by_source


def camera_connection_state(state, now: float | None = None) -> dict:
    """Derive camera state only from a real WebRTC or processed-frame observation."""
    now = time.time() if now is None else now
    metrics = state.get("last_detection_metrics") or {}
    metrics_time = metrics.get("updated_at")
    recent_browser_frame = (
        metrics.get("source") == "Browser webcam"
        and isinstance(metrics_time, (int, float))
        and 0 <= now - metrics_time <= 5
    )
    stream_active = bool(state.get("webcam_stream_active"))
    return {
        "browser_status": "Connected" if stream_active else "Not connected",
        "browser_inference": bool(stream_active and recent_browser_frame),
        "browser_last_frame": (
            time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(metrics_time))
            if metrics.get("source") == "Browser webcam" and metrics_time
            else "Unavailable"
        ),
        "local_inference": bool(
            state.get("detection_running")
            and state.get("selected_source") == "Webcam"
        ),
    }


def _latest_incident_time(source: str) -> str:
    try:
        rows = fetch_recent_alerts_by_source(source, limit=1)
    except Exception:
        return "Unavailable"
    if not rows:
        return "None recorded"
    return format_saved_timestamp_local(rows[0]["timestamp"])


def render_camera_sources() -> None:
    state = st.session_state
    snapshot = camera_connection_state(state)
    browser_incident_time = _latest_incident_time("Browser webcam")
    local_source = f"Webcam {CAMERA_SOURCE} (local camera)"
    local_metrics = state.get("last_detection_metrics") or {}
    local_frame_time = "Unavailable"
    if local_metrics.get("source", "").startswith("Webcam ") and local_metrics.get("updated_at"):
        local_frame_time = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(local_metrics["updated_at"]))

    st.subheader("Camera source status")
    st.caption("This app supports one browser stream at a time and bounded local-camera video analysis. It does not currently ingest RTSP/IP streams or run multi-camera inference.")
    columns = st.columns(2)
    with columns[0]:
        with st.container(border=True):
            st.markdown("#### Browser webcam")
            st.metric("Connection", snapshot["browser_status"])
            st.write(f"**Inference active:** {'Yes' if snapshot['browser_inference'] else 'No'}")
            st.write(f"**Last successful frame:** {snapshot['browser_last_frame']}")
            st.write(f"**Latest incident:** {browser_incident_time}")
            if not state.get("webcam_stream_active"):
                st.caption("To connect, open Live Detection and start the embedded browser camera. Camera permission and HTTPS/localhost are required.")
            elif not snapshot["browser_inference"]:
                st.caption("The WebRTC stream is active; no processed frame has succeeded in the last five seconds.")
    with columns[1]:
        with st.container(border=True):
            st.markdown("#### Local camera")
            st.metric("Mode", "On-demand Video Analysis")
            st.write(f"**Source index:** {CAMERA_SOURCE}")
            st.write(f"**Inference active:** {'Yes' if snapshot['local_inference'] else 'No'}")
            st.write(f"**Last processed frame:** {local_frame_time}")
            st.write(f"**Latest incident:** {_latest_incident_time(local_source)}")
            st.caption("This camera is opened only when a bounded Video Analysis run is started on the machine hosting Streamlit. It is not a persistent camera connection.")
            if state.get("selected_source") == "Webcam" and state.get("last_processing_error"):
                st.warning("The last local-camera run could not open a readable stream. Check host camera permissions and whether another application is using the camera, then retry Video Analysis.")
    st.info("RTSP/IP cameras are unsupported in this build. No credentials are collected, and no Connect button is shown for an unavailable ingestion path.")
