"""Simple analytics cards for the dashboard."""

import streamlit as st

from database.database import count_alerts


def render_statistics() -> None:
    """Render database-backed analytics cards."""
    st.subheader("Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Alerts", count_alerts())
    col2.metric("Active Alerts", count_alerts(status="Active"))
    col3.metric("Critical Alerts", count_alerts(status="Active", severity="Critical"))
