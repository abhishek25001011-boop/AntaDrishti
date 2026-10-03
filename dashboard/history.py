"""Database-backed incident history and alert center."""

from pathlib import Path

import pandas as pd
import streamlit as st

from database.database import fetch_alerts

INCIDENT_COLUMNS = ["id", "event_type", "severity", "message", "source", "timestamp", "status", "snapshot_path"]


def incidents_frame() -> pd.DataFrame:
    rows = fetch_alerts(limit=None)
    frame = pd.DataFrame([dict(row) for row in rows], columns=INCIDENT_COLUMNS)
    for column in ("event_type", "severity", "message", "source", "timestamp", "status", "snapshot_path"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def _reset_filters() -> None:
    st.session_state["history_event_filter"] = "All events"
    st.session_state["history_severity_filter"] = "All severities"
    st.session_state["history_source_filter"] = "All sources"


def _filtered_history(frame: pd.DataFrame) -> pd.DataFrame:
    controls = st.columns([1, 1, 1, 0.8])
    events = ["All events", *sorted(frame["event_type"].dropna().astype(str).unique())]
    severities = ["All severities", "High", "Medium", "Low"]
    source_values = frame["source"].dropna().astype(str).unique()
    sources = ["All sources", *sorted(value for value in source_values if value)]
    if "" in source_values:
        sources.append("Not recorded")
    event = controls[0].selectbox("Event type", events, key="history_event_filter")
    severity = controls[1].selectbox("Severity", severities, key="history_severity_filter")
    source = controls[2].selectbox("Source", sources, key="history_source_filter")
    controls[3].button("Reset filters", on_click=_reset_filters, width="stretch")

    filtered = frame
    if event != "All events":
        filtered = filtered[filtered.event_type == event]
    if severity != "All severities":
        filtered = filtered[filtered.severity == severity]
    if source == "Not recorded":
        filtered = filtered[filtered.source == ""]
    elif source != "All sources":
        filtered = filtered[filtered.source == source]
    return filtered


def _render_incident_details(frame: pd.DataFrame) -> None:
    st.markdown("#### Incident details")
    if frame.empty:
        st.info("No incidents match the current filters.")
        return
    rows = {int(row.id): row for row in frame.itertuples(index=False)}
    incident_id = st.selectbox(
        "Select an incident", list(rows),
        format_func=lambda value: f"#{value} · {rows[value].event_type} · {rows[value].timestamp}",
        label_visibility="collapsed",
    )
    item = rows[incident_id]
    with st.container(border=True):
        st.markdown(f"**{item.event_type}** · **{item.severity}**")
        st.write(f"Timestamp: {item.timestamp}")
        st.write(f"Source: {item.source or 'Not recorded'}")
        if item.message:
            st.write(item.message)
        if item.snapshot_path:
            snapshot = Path(item.snapshot_path)
            if snapshot.is_file():
                st.image(str(snapshot), caption=f"Incident #{item.id} snapshot", width=640)
            else:
                st.caption("Snapshot was recorded but is not available at its saved path.")
        else:
            st.caption("No snapshot is associated with this incident.")


def render_alert_center() -> None:
    st.subheader("Alert Center")
    frame = incidents_frame()
    if frame.empty:
        st.info("No incidents recorded yet.")
        st.download_button(
            "Export Incident History", data="", file_name="antadrishti_incidents.csv",
            mime="text/csv", disabled=True,
        )
        return

    filtered = _filtered_history(frame)
    csv_data = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Incident History", data=csv_data,
        file_name="antadrishti_incidents.csv", mime="text/csv",
    )
    st.caption(f"Showing {len(filtered)} of {len(frame)} stored incidents. Export includes all stored incidents.")
    if filtered.empty:
        st.info("No incidents match the selected filters.")
    else:
        view = filtered[INCIDENT_COLUMNS].drop(columns=["message", "snapshot_path"])
        st.dataframe(view, width="stretch", hide_index=True)
        st.markdown("#### Recent alerts")
        for row in filtered.head(5).itertuples(index=False):
            with st.container(border=True):
                color = {"High": "high", "Medium": "medium", "Low": "low"}.get(row.severity, "low")
                st.markdown(f"**{row.event_type}** · <span class='ad-severity-{color}'>{row.severity}</span>", unsafe_allow_html=True)
                st.caption(f"{row.timestamp} · {row.source or 'Source not recorded'}")
    _render_incident_details(filtered)


def render_recent_alerts(limit: int = 5) -> None:
    st.markdown("#### Recent incidents")
    frame = incidents_frame().head(limit)
    if frame.empty:
        st.info("No incidents recorded yet.")
        return
    for row in frame.itertuples(index=False):
        with st.container(border=True):
            color = {"High": "high", "Medium": "medium", "Low": "low"}.get(row.severity, "low")
            st.markdown(f"**{row.event_type}** · <span class='ad-severity-{color}'>{row.severity}</span>", unsafe_allow_html=True)
            st.caption(f"{row.timestamp} · {row.source or 'Source not recorded'}")
