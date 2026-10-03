"""Incident totals and database-backed activity charts."""

import pandas as pd
import streamlit as st

from database.database import count_alerts, count_alerts_last_hours
from dashboard.history import incidents_frame


def render_statistics() -> None:
    st.subheader("Incident analytics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total incidents", count_alerts())
    col2.metric("High severity", count_alerts(severity="High"))
    col3.metric("Alerts · 24 hours", count_alerts_last_hours(24))


def render_analytics_page() -> None:
    st.subheader("Analytics")
    frame = incidents_frame()
    if frame.empty:
        st.info("No incidents recorded yet.")
        return

    render_statistics()

    a, b = st.columns(2)
    with a:
        st.markdown("#### Incidents by event type")
        event_counts = frame.groupby("event_type").size().sort_values(ascending=False)
        st.bar_chart(event_counts, horizontal=True, color="#13bdd6")
    with b:
        st.markdown("#### Incidents by severity")
        severity_counts = frame.groupby("severity").size().reindex(
            ["Low", "Medium", "High"], fill_value=0
        )
        st.bar_chart(severity_counts, color="#18b6d1")

    st.markdown("#### Recent incident activity")
    times = pd.to_datetime(frame["timestamp"], errors="coerce")
    activity = (
        frame.assign(day=times.dt.floor("D"))
        .dropna(subset=["day"])
        .groupby("day")
        .size()
        .rename("Incidents")
    )
    if activity.empty:
        st.caption("Stored timestamps could not be grouped into daily activity.")
    else:
        st.line_chart(activity, color="#13bdd6")
