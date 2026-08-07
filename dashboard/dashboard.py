"""Main dashboard layout for the Streamlit app."""

import streamlit as st

from dashboard.analytics import render_statistics
from dashboard.history import render_recent_alerts
from dashboard.live_feed import render_live_detection
from dashboard.sidebar import render_sidebar
from database.database import count_alerts
from utils.helpers import format_alert_count


def render_dashboard() -> None:
    """Render the full dashboard UI."""
    render_sidebar()

    st.title("🛡️ ANTAHDRISHTI")
    st.caption("AI-Powered Public Safety Intelligence System")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Camera Feed")
        render_live_detection()

    with col2:
        st.subheader("Alert Panel")
        active_alerts = count_alerts(status="Active")
        critical_alerts = count_alerts(status="Active", severity="Critical")
        st.warning("Monitoring in progress.")
        st.metric("Active Alerts", format_alert_count(active_alerts))
        st.metric("Critical Events", format_alert_count(critical_alerts))

    render_statistics()
    render_recent_alerts()

    st.markdown("---")
    st.caption("Prototype build · Not for production use")
