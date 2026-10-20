"""Database-backed incident history and alert center."""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import cv2

from database.database import fetch_alerts
from config import SNAPSHOT_DIR

logger = logging.getLogger(__name__)

INCIDENT_COLUMNS = ["id", "event_type", "severity", "message", "source", "timestamp", "status", "snapshot_path", "metadata"]


def incidents_frame() -> pd.DataFrame:
    rows = fetch_alerts(limit=None)
    frame = pd.DataFrame([dict(row) for row in rows], columns=INCIDENT_COLUMNS)
    for column in ("event_type", "severity", "message", "source", "timestamp", "status", "snapshot_path", "metadata"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def _reset_filters() -> None:
    st.session_state["history_event_filter"] = "All events"
    st.session_state["history_severity_filter"] = "All severities"
    st.session_state["history_source_filter"] = "All sources"
    st.session_state["history_search"] = ""


def _source_label(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return "Not recorded"
    normalized = value.replace("\\", "/")
    if "/" in normalized:
        return Path(normalized).name or "Recorded media"
    return value


def _filtered_history(frame: pd.DataFrame) -> pd.DataFrame:
    controls = st.columns([1, 1, 1, 1.25, 0.8])
    events = ["All events", *sorted(frame["event_type"].dropna().astype(str).unique())]
    severities = ["All severities", "High", "Medium", "Low"]
    source_values = frame["source"].dropna().astype(str).unique()
    source_labels = {}
    label_counts = {}
    for value in source_values:
        if not value:
            continue
        label = _source_label(value)
        label_counts[label] = label_counts.get(label, 0) + 1
        if label_counts[label] > 1:
            label = f"{label} ({label_counts[label]})"
        source_labels[label] = value
    sources = ["All sources", *sorted(source_labels)]
    if "" in source_values:
        sources.append("Not recorded")
    event = controls[0].selectbox("Event type", events, key="history_event_filter")
    severity = controls[1].selectbox("Severity", severities, key="history_severity_filter")
    source = controls[2].selectbox("Source", sources, key="history_source_filter")
    search = controls[3].text_input("Search incidents", key="history_search", placeholder="ID, event, source")
    controls[4].button("Reset", on_click=_reset_filters, width="stretch")

    filtered = frame
    if event != "All events":
        filtered = filtered[filtered.event_type == event]
    if severity != "All severities":
        filtered = filtered[filtered.severity == severity]
    if source == "Not recorded":
        filtered = filtered[filtered.source == ""]
    elif source != "All sources":
        filtered = filtered[filtered.source == source_labels[source]]
    if search.strip():
        query = search.strip().casefold()
        searchable = filtered.astype(str).apply(lambda column: column.str.casefold())
        filtered = filtered[searchable.apply(lambda column: column.str.contains(query, regex=False)).any(axis=1)]
    return filtered


def _local_timestamp(value: str) -> str:
    """Present saved UTC timestamps in local time, preserving unknown values honestly."""
    if not value:
        return "Unavailable"
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return "Unavailable"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _safe_snapshot(snapshot_path: str):
    """Read a snapshot only when it stays under the configured folder and is valid."""
    if not snapshot_path:
        return None
    try:
        root = SNAPSHOT_DIR.resolve()
        candidate = Path(snapshot_path).resolve()
        candidate.relative_to(root)
        if not candidate.is_file():
            return None
        image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except (OSError, RuntimeError, ValueError):
        return None


def _render_incident_details(frame: pd.DataFrame) -> None:
    st.markdown("#### Incident details")
    if frame.empty:
        st.info("No incidents match the current filters.")
        return
    rows = {int(row.id): row for row in frame.itertuples(index=False)}
    incident_id = st.selectbox(
        "Select an incident", list(rows),
        format_func=lambda value: f"#{value} | {rows[value].event_type} | {_local_timestamp(rows[value].timestamp)}",
        label_visibility="collapsed",
    )
    item = rows[incident_id]
    with st.container(border=True):
        st.markdown(f"**{item.event_type} | {item.severity}**")
        st.write(f"Timestamp: {_local_timestamp(item.timestamp)}")
        st.write(f"Source: {_source_label(item.source)}")
        if item.message:
            st.write(item.message)
        if item.metadata:
            try:
                st.json(json.loads(item.metadata), expanded=False)
            except (TypeError, json.JSONDecodeError):
                st.code(item.metadata)
        if item.snapshot_path:
            snapshot = _safe_snapshot(item.snapshot_path)
            if snapshot is not None:
                st.image(snapshot, caption=f"Incident #{item.id} snapshot", width="stretch")
            else:
                st.caption("Snapshot is missing or unavailable.")
        else:
            st.caption("No snapshot is associated with this incident.")


def render_alert_center() -> None:
    st.subheader("Alert Center")
    try:
        frame = incidents_frame()
    except Exception:
        logger.exception("Could not load incident history")
        st.error("Incident history is temporarily unavailable. Check the database connection in Settings.")
        return
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
    st.caption(
        f"Showing {len(filtered)} of {len(frame)} stored incidents. "
        "Times are shown in local time; CSV preserves saved timestamps."
    )
    if filtered.empty:
        st.info("No incidents match the selected filters.")
    else:
        view = filtered[INCIDENT_COLUMNS].drop(columns=["message", "snapshot_path", "metadata"]).copy()
        view["timestamp"] = view["timestamp"].map(_local_timestamp)
        view = view.rename(columns={
            "id": "Incident ID", "event_type": "Event type", "severity": "Severity",
                "source": "Source", "timestamp": "Local time", "status": "Status",
        })
        view["Source"] = view["Source"].map(_source_label)
        st.dataframe(view, width="stretch", hide_index=True)
        st.markdown("#### Recent alerts")
        for row in filtered.head(5).itertuples(index=False):
            with st.container(border=True):
                color = {"High": "high", "Medium": "medium", "Low": "low"}.get(row.severity, "low")
                st.markdown(
                    f"**{row.event_type} | <span class='ad-severity-{color}'>{row.severity}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"{_local_timestamp(row.timestamp)} | {_source_label(row.source)}")
    _render_incident_details(filtered)

def render_recent_alerts(limit: int = 5) -> None:
    st.markdown("#### Recent incidents")
    try:
        rows = fetch_alerts(limit=max(0, int(limit)))
    except Exception:
        logger.exception("Could not load recent incidents")
        st.info("Recent incident history is temporarily unavailable.")
        return
    if not rows:
        st.info("No incidents recorded yet.")
        return
    for row in rows:
        item = dict(row)
        color = {"High": "high", "Medium": "medium", "Low": "low"}.get(item["severity"], "low")
        with st.container(border=True):
            st.markdown(
                f"**{item['event_type']}** | <span class='ad-severity-{color}'>{item['severity']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"{_local_timestamp(item['timestamp'])} | {_source_label(item['source'])}")
