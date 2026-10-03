"""Page routing and overview for the public-safety command center."""

import streamlit as st

from config import APP_DESCRIPTION
from dashboard.analytics import render_analytics_page
from dashboard.history import render_alert_center, render_recent_alerts
from dashboard.live_feed import render_live_detection
from dashboard.sidebar import render_sidebar
from dashboard.system_info import application_status, render_settings
from dashboard.theme import apply_theme, render_header
from database.database import count_alerts, count_alerts_last_hours
from utils.video_utils import list_sample_videos


def _render_overview() -> None:
    st.markdown("<div class='ad-kicker'>Operations overview</div>", unsafe_allow_html=True)
    st.title("Safety monitoring dashboard")
    st.caption(APP_DESCRIPTION)

    sample_count = len(list_sample_videos())
    source_modes = 2 + sample_count  # local camera + upload mode + listed sample files
    kpis = st.columns(4)
    kpis[0].metric("Available Sources", f"{source_modes}", help="Input modes plus nonempty sample video files.")
    kpis[1].metric("Total Incidents", count_alerts())
    kpis[2].metric("High Severity", count_alerts(severity="High"))
    kpis[3].metric("Alerts · 24 hours", count_alerts_last_hours(24))
    st.caption(f"{sample_count} valid sample video(s) · webcam availability depends on this host · uploads accepted on demand")

    left, right = st.columns([1.7, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### Detection workspace")
            frame = st.session_state.get("last_frame_rgb")
            if frame is not None:
                st.image(frame, caption="Most recently processed frame", width="stretch")
            else:
                st.info("No frame processed in this session yet. Open Live Detection or Video Analysis to begin.")
            summary = st.session_state.get("last_summary")
            if summary:
                st.caption(
                    f"Last run: {summary.frames_processed} frame(s) · "
                    f"{len(summary.incidents)} incident(s) · "
                    f"source: {st.session_state.get('last_run_source', 'not recorded')}"
                )
    with right:
        with st.container(border=True):
            st.markdown("#### System status")
            st.markdown(f"**{application_status()}**")
            st.caption("Person detection uses the included YOLOv8n COCO model.")
            st.caption("Fall and possible altercation alerts use experimental heuristics.")
            st.caption("No incident counts or validation metrics are estimated.")

    render_recent_alerts(limit=4)


def render_dashboard() -> None:
    apply_theme()
    page = render_sidebar()
    render_header(application_status())

    if page == "Dashboard":
        _render_overview()
    elif page == "Live Detection":
        st.title("Live Detection")
        st.caption("One bounded run per Start action. Camera input uses the host machine's local camera.")
        render_live_detection()
    elif page == "Video Analysis":
        st.title("Video Analysis")
        st.caption("Analyze one selected sample or uploaded clip through the shared detection pipeline.")
        render_live_detection()
    elif page == "Alerts":
        st.title("Incident History & Alert Center")
        render_alert_center()
    elif page == "Analytics":
        st.title("Incident Analytics")
        render_analytics_page()
    elif page == "Settings":
        st.title("System Configuration")
        render_settings()
