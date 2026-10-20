"""Page routing and data-backed overview for the public-safety command center."""

import logging
import time

import streamlit as st

from config import APP_DESCRIPTION
from config import YOLO_MODEL_PATH
from dashboard.alert_controls import (
    initialize_alert_state,
    render_alert_preferences,
    render_new_incident_alerts,
)
from dashboard.analytics import render_analytics_page
from dashboard.camera_sources import render_camera_sources
from dashboard.history import _source_label, render_alert_center, render_recent_alerts
from dashboard.live_feed import render_browser_webcam, render_live_detection
from dashboard.sidebar import render_sidebar
from dashboard.system_info import application_status, render_settings
from dashboard.theme import apply_theme, render_header, render_page_heading
from database.database import get_incident_summary
from utils.video_utils import list_sample_videos

logger = logging.getLogger(__name__)


def _navigate_to(page: str) -> None:
    st.session_state.navigation = page


def _render_overview() -> None:
    render_page_heading("Operations overview", "Command center", APP_DESCRIPTION)

    sample_count = len(list_sample_videos())
    try:
        summary = get_incident_summary()
    except Exception:
        logger.exception("Could not load command-center incident totals")
        summary = None
        st.warning("Incident totals are temporarily unavailable. Check the database connection in Settings.")
    if summary is not None:
        kpis = st.columns(5)
        kpis[0].metric("Active incidents", summary["active"])
        kpis[1].metric("High severity", summary["high"])
        kpis[2].metric("Medium severity", summary["medium"])
        kpis[3].metric("Low severity", summary["low"])
        kpis[4].metric("Incidents · 24 hours", summary["last_24h"])
        st.caption(
            f"{summary['total']} incidents recorded · {sample_count} nonempty sample video(s) available. "
            "Incident totals come from SQLite history."
        )
    else:
        st.caption(f"{sample_count} nonempty sample video(s) available.")

    shortcuts = st.columns(4)
    for column, page in zip(shortcuts, ("Live Detection", "Video Analysis", "Alerts", "Analytics")):
        column.button(
            page,
            width="stretch",
            key=f"overview_{page.lower().replace(' ', '_')}",
            on_click=_navigate_to,
            args=(page,),
        )

    left, right = st.columns([1.65, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### Detection workspace")
            frame = st.session_state.get("last_frame_rgb")
            if frame is not None:
                st.image(frame, caption="Most recently processed frame", width="stretch")
            else:
                st.info("No frame has been processed in this session yet. Start Live Detection or Video Analysis to begin.")

            metrics = st.session_state.get("last_detection_metrics")
            if metrics:
                age = max(0, int(time.time() - metrics["updated_at"]))
                if age <= 5:
                    age_label = f"{age}s ago"
                elif age < 60:
                    age_label = f"{age}s ago · stale"
                else:
                    age_label = f"{age // 60}m ago · stale"
                readings = st.columns(3)
                readings[0].metric("People · last processed", metrics["people"])
                readings[1].metric("Crowd estimate", metrics["crowd"])
                readings[2].metric("Inference FPS", f"{metrics['fps']:.1f}")
                st.caption(f"Last processed {age_label} · source: {metrics['source']}")

            summary_state = st.session_state.get("last_summary")
            if summary_state:
                st.caption(
                    f"Last video run: {summary_state.frames_processed} frame(s) · "
                    f"{len(summary_state.incidents)} incident(s) · "
                    f"source: {_source_label(st.session_state.get('last_run_source', 'not recorded'))}"
                )
    with right:
        with st.container(border=True):
            st.markdown("#### Operational status")
            st.markdown(f"**Application:** {application_status()}")
            model_available = YOLO_MODEL_PATH.is_file() and YOLO_MODEL_PATH.stat().st_size > 0
            manager = st.session_state.get("detector_manager")
            model_loaded = manager is not None and getattr(manager, "_model", None) is not None
            if model_loaded:
                model_state = "Loaded in this session"
            elif model_available:
                model_state = "Installed · loads on first inference"
            else:
                model_state = "Model file missing"
            st.caption(f"Detection model: {model_state}")
            camera_state = "Browser camera streaming" if st.session_state.get("webcam_stream_active") else "Browser camera idle"
            st.caption(camera_state)
            st.caption("Fall and possible altercation signals are experimental heuristics.")
            st.caption("Saved incident totals are shown above; no counts or validation metrics are estimated.")

    render_recent_alerts(limit=5)


def render_dashboard() -> None:
    initialize_alert_state(st.session_state)
    page = render_sidebar()
    apply_theme(st.session_state.theme_mode)
    render_header(application_status())

    if page == "Dashboard":
        _render_overview()
    elif page == "Live Detection":
        render_page_heading(
            "Live monitoring",
            "Browser camera",
            "Allow camera access when prompted. WebRTC requires HTTPS or localhost.",
        )
        render_browser_webcam()
    elif page == "Video Analysis":
        render_page_heading(
            "Recorded media",
            "Video analysis",
            "Run bounded detection on an uploaded clip or one of the included sample videos.",
        )
        render_live_detection()
    elif page == "Camera Sources":
        render_page_heading(
            "Input inventory", "Camera sources", "Review actual camera workflow status and supported connection types."
        )
        render_camera_sources()
    elif page == "Alerts":
        render_page_heading(
            "Operations records", "Alerts and history", "Review saved incidents, snapshots, and event details."
        )
        render_alert_center()
    elif page == "Analytics":
        render_page_heading(
            "Operations insight", "Incident analytics", "Explore trends computed from saved incident history."
        )
        render_analytics_page()
    elif page == "Settings":
        render_page_heading(
            "System configuration", "Settings", "Review runtime status, alert controls, and session preferences."
        )
        render_settings()
        render_alert_preferences()

    if page != "Live Detection":
        render_new_incident_alerts()
