"""Database-backed incident totals and activity charts."""

import logging

import pandas as pd
import streamlit as st

from dashboard.history import _source_label, incidents_frame
from database.database import get_incident_summary

logger = logging.getLogger(__name__)


def render_statistics() -> None:
    st.subheader("Persisted incident totals")
    try:
        summary = get_incident_summary()
    except Exception:
        logger.exception("Could not load incident totals")
        st.error("Incident totals are temporarily unavailable. Check the database connection in Settings.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total incidents", summary["total"])
    col2.metric("Active incidents", summary["active"])
    col3.metric("High severity", summary["high"])
    col4.metric("Last 24 hours", summary["last_24h"])


def render_analytics_page() -> None:
    st.subheader("Incident analytics")
    try:
        frame = incidents_frame()
    except Exception:
        logger.exception("Could not load analytics history")
        st.error("Analytics are temporarily unavailable. Check the database connection in Settings.")
        return
    if frame.empty:
        st.info("No incidents have been persisted yet. Analytics will appear after an event is saved.")
        return

    render_statistics()
    frame["event_type"] = frame["event_type"].replace("", "Unknown event")
    frame["severity"] = frame["severity"].replace("", "Low")
    frame["source"] = frame["source"].map(_source_label)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Incidents by event type")
        event_counts = frame.groupby("event_type").size().sort_values(ascending=True).tail(12)
        st.bar_chart(event_counts, horizontal=True, color="#42bfd2")
    with right:
        st.markdown("#### Incidents by severity")
        severity_counts = frame.groupby("severity").size().reindex(
            ["Low", "Medium", "High"], fill_value=0
        )
        st.bar_chart(severity_counts, color="#42bfd2")

    st.markdown("#### Incidents by source")
    sources = frame["source"].replace("", "Not recorded").fillna("Not recorded")
    source_counts = sources.value_counts().sort_values(ascending=True).tail(10)
    st.bar_chart(source_counts, horizontal=True, color="#42bfd2")
    st.caption("Source is recorded by the workflow that saved each incident.")

    st.markdown("#### Incident activity over time")
    times = pd.to_datetime(frame["timestamp"], errors="coerce")
    activity = (
        frame.assign(day=times.dt.floor("D"))
        .dropna(subset=["day"])
        .groupby("day")
        .size()
        .rename("Incidents")
    )
    if activity.empty:
        st.caption("Stored timestamps are unavailable or could not be parsed.")
    else:
        st.line_chart(activity, color="#42bfd2")
    st.caption("Charts summarize persisted SQLite incidents; they do not predict future events.")
