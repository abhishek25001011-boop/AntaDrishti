"""Bounded, user-started processing for the selected video or camera source."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from config import ALERT_COOLDOWN_SECONDS, CAMERA_SOURCE, MAX_PROCESSING_FRAMES, SNAPSHOT_DIR, YOLO_CONFIDENCE_THRESHOLD
from database.database import count_alerts, fetch_alerts_by_ids, fetch_recent_alerts_by_source
from detectors.detection_manager import DetectionManager
from utils.alert_cooldown import AlertCooldown
from utils.event_detection import EventDetector
from utils.video_utils import open_video_capture, release_capture
from utils.video_pipeline import process_capture
from utils.webcam_pipeline import WebcamProcessor, WebcamUpdate

logger = logging.getLogger(__name__)


def _open_video_analysis() -> None:
    st.session_state.navigation = "Video Analysis"


def init_detection_state() -> None:
    defaults = {
        "detection_running": False,
        "video_capture": None,
        "capture_source": None,
        "detector_manager": None,
        "alert_cooldown": None,
        "selected_source": "Webcam",
        "last_summary": None,
        "last_video_frame_rgb": None,
        "last_processing_error": None,
        "webcam_processor": None,
        "webcam_last_update": None,
        "webcam_incidents": [],
        "webcam_session_incident_ids": [],
        "webcam_pending_incidents": [],
        "webcam_persist_error": None,
        "last_detection_metrics": None,
        "webcam_stream_active": False,
        "webcam_session_started_at": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_live_detection() -> None:
    """Process one bounded run and then return control to Streamlit."""
    init_detection_state()
    video_area = st.empty()
    status_area = st.empty()

    if not st.session_state.detection_running:
        error = st.session_state.get("last_processing_error")
        if error and not st.session_state.last_summary:
            status_area.error(error)
        if st.session_state.last_summary:
            summary = st.session_state.last_summary
            _render_video_summary(summary, st.session_state.get("last_run_source", "Unknown source"))
        elif not error:
            video_area.info(
                "Choose a source in the sidebar and press Start Detection. "
                "This page analyzes uploaded or sample videos; browser webcam results are shown under Live Detection."
            )
            st.caption(
                "Persisted incidents from either workflow are available under Alerts and History; "
                "Analytics summarizes all saved incidents."
            )
        return

    source = st.session_state.selected_source
    uploaded_file = st.session_state.get("selected_uploaded_video")
    sample_video_path = st.session_state.get("selected_sample_video_path")
    confidence = st.session_state.get("conf_threshold", YOLO_CONFIDENCE_THRESHOLD)

    if source == "Sample Video":
        event_source = sample_video_path or source
    elif source == "Upload Video":
        event_source = getattr(uploaded_file, "name", source)
    else:
        event_source = f"Webcam {CAMERA_SOURCE} (local camera)"
    display_source = Path(event_source.replace("\\", "/")).name
    st.session_state.last_run_source = event_source
    st.session_state.last_summary = None
    st.session_state.last_processing_error = None
    st.session_state.last_video_frame_rgb = None

    capture = None
    try:
        capture = open_video_capture(source, uploaded_file, sample_video_path)
        if capture is None or not capture.isOpened():
            if source == "Webcam":
                message = "Could not access the local webcam. Check camera permissions and that another app is not using it."
            else:
                message = "Could not open this video. Check that the selected file is valid and nonempty."
            st.session_state.last_processing_error = message
            status_area.error(message)
            return

        if st.session_state.detector_manager is None:
            st.session_state.detector_manager = DetectionManager()
        if st.session_state.alert_cooldown is None:
            st.session_state.alert_cooldown = AlertCooldown(
                st.session_state.get("alert_cooldown_seconds", ALERT_COOLDOWN_SECONDS)
            )

        progress_area = st.progress(0.0, text="Starting video processing…")
        event_detector = EventDetector()
        from dashboard.alert_controls import register_persisted_incident

        def update_frame(index, limit, total, result, incidents):
            target = min(total, limit) if total else limit
            for incident in incidents:
                register_persisted_incident(st.session_state, incident)
            progress_area.progress(min(1.0, index / max(1, target)), text=f"Processing frame {index} of {target} | source: {display_source}")
            rgb_frame = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
            st.session_state.last_video_frame_rgb = rgb_frame
            st.session_state.last_frame_rgb = rgb_frame
            st.session_state.last_detection_metrics = {
                "people": result.person_count,
                "crowd": result.crowd_density,
                "fps": result.fps,
                "source": display_source,
                "updated_at": time.time(),
            }
            video_area.image(rgb_frame, channels="RGB", width="stretch")
            status_area.write(
                f"Detection Active | Source: {display_source} | Frame: {index} | FPS: {result.fps:.1f} | "
                f"People: {result.person_count} | Bags: {result.bag_count} | "
                f"Crowd: {result.crowd_density} ({result.crowd_occupancy:.0%} frame occupancy)"
            )
            for incident in incidents:
                st.warning(f"{incident['event_type']} · {incident['severity']} · {incident['message']}")

        summary = process_capture(
            capture=capture,
            source=event_source,
            manager=st.session_state.detector_manager,
            event_detector=event_detector,
            cooldown=st.session_state.alert_cooldown,
            conf_threshold=confidence,
            frame_limit=MAX_PROCESSING_FRAMES,
            on_frame=update_frame,
        )
        st.session_state.last_summary = summary
        st.session_state.last_processing_error = summary.error
        if summary.error:
            status_area.error(summary.error)
        elif summary.stopped:
            status_area.info(f"Processing stopped after {summary.frames_processed} frame(s).")
        elif summary.frames_processed >= MAX_PROCESSING_FRAMES and not summary.reached_end:
            status_area.info(
                f"Safety limit reached: processed {summary.frames_processed} frames. "
                "Press Start Detection to process the next run from the beginning."
            )
        else:
            status_area.success(
                f"Processing complete: {summary.frames_processed} frame(s), "
                f"{len(summary.incidents)} new incident(s)."
            )
        st.caption(
            f"Active alerts: {count_alerts(status='Active')} | "
            f"High severity: {count_alerts(severity='High')}"
        )
    except Exception:
        logger.exception("Video analysis failed")
        st.session_state.last_processing_error = "Video analysis failed."
        status_area.error("Video analysis failed. Check the selected input and try again.")
    finally:
        release_capture(capture)
        st.session_state.video_capture = None
        st.session_state.capture_source = None
        st.session_state.detection_running = False


def _render_video_summary(summary, source: str) -> None:
    """Render retained results from the last uploaded or sample-video run."""
    st.subheader("Analysis results")
    metrics = st.columns(3)
    metrics[0].metric("Frames processed", summary.frames_processed)
    metrics[1].metric("Peak people in one frame", summary.peak_people_count)
    metrics[2].metric("Incidents", len(summary.incidents))
    display_source = Path(source).name if source not in {"Upload Video", "Webcam", "Unknown source"} else source
    st.caption(f"Recorded-video workflow | Source: {display_source}")
    if summary.error:
        st.error(summary.error)
        return
    if summary.stopped:
        st.info(f"Processing stopped after {summary.frames_processed} frame(s).")
    elif summary.frames_processed >= summary.frame_limit and not summary.reached_end:
        st.info(
            f"Safety limit reached after {summary.frames_processed} frame(s); "
            "the selected video may have more frames."
        )
    elif not summary.incidents:
        st.success("Analysis completed - no incidents detected")
    else:
        st.success(f"Analysis completed - {len(summary.incidents)} incident(s) detected and saved.")

    frame = st.session_state.get("last_video_frame_rgb")
    if frame is not None:
        with st.container(border=True):
            st.markdown("#### Latest annotated frame")
            st.image(frame, caption="Last processed frame from this video run", width="stretch")
    if not summary.incidents:
        return
    try:
        rows = fetch_alerts_by_ids(item.get("id") for item in summary.incidents)
    except Exception:
        logger.exception("Could not load video analysis incident details")
        st.warning("Incident details could not be loaded from history. The analysis itself completed.")
        return

    incidents = [dict(row) for row in rows]
    if not incidents:
        st.warning("No persisted incident details were found for this analysis run.")
        return
    snapshots = {int(row["id"]): _load_incident_snapshot(row["snapshot_path"]) for row in incidents}
    st.markdown("#### Incident results")
    st.dataframe(
        [
            {
                "Incident ID": row["id"],
                "Event type": row["event_type"],
                "Severity": row["severity"],
                "Local time": format_saved_timestamp_local(row["timestamp"]),
                "Snapshot": "Available" if snapshots[int(row["id"])][0] is not None else "Unavailable",
            }
            for row in incidents
        ],
        hide_index=True,
        width="stretch",
    )
    st.markdown("#### Incident snapshots")
    for row in incidents:
        image, note = snapshots[int(row["id"])]
        with st.container(border=True):
            st.markdown(f"**#{row['id']} | {row['event_type']} | {row['severity']}**")
            st.caption(format_saved_timestamp_local(row["timestamp"]))
            if image is not None:
                st.image(image, caption="Saved incident snapshot", width="stretch")
            else:
                st.caption(note or "No saved snapshot is available.")

def _parse_saved_timestamp(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def format_saved_timestamp_local(value: str | None) -> str:
    """Format the stored UTC timestamp in local time without inventing a value."""
    parsed = _parse_saved_timestamp(value)
    if parsed is None:
        return "Unavailable"
    local_time = parsed.astimezone()
    format_string = "%Y-%m-%d %H:%M:%S.%f %z" if parsed.microsecond else "%Y-%m-%d %H:%M:%S %z"
    return local_time.strftime(format_string)


def _incident_description(record) -> str:
    message = record.get("message") if hasattr(record, "get") else None
    if isinstance(message, str) and message.strip():
        return message.strip()
    raw_metadata = record.get("metadata") if hasattr(record, "get") else None
    if raw_metadata:
        try:
            metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
        except (TypeError, ValueError):
            metadata = None
        if isinstance(metadata, dict):
            description = metadata.get("description")
            if isinstance(description, str) and description.strip():
                return description.strip()
    return "Unavailable"


def _partition_live_incidents(records, current_session_ids):
    """Deduplicate persisted rows, order newest first, and label this app session."""
    unique = {}
    for item in records:
        record = dict(item)
        incident_id = record.get("id")
        if incident_id is not None:
            unique[int(incident_id)] = record

    def sort_key(record):
        parsed = _parse_saved_timestamp(record.get("timestamp"))
        return (parsed.timestamp() if parsed is not None else float("-inf"), int(record["id"]))

    ordered = sorted(unique.values(), key=sort_key, reverse=True)
    current_ids = {int(value) for value in current_session_ids if value is not None}
    current = [record for record in ordered if int(record["id"]) in current_ids]
    previous = [record for record in ordered if int(record["id"]) not in current_ids]
    return current, previous


def _load_incident_snapshot(snapshot_path: str | None):
    """Read an image only from the configured snapshot directory; hide all paths."""
    if not snapshot_path:
        return None, "No snapshot was saved."
    try:
        root = SNAPSHOT_DIR.resolve()
        candidate = Path(snapshot_path).resolve()
        candidate.relative_to(root)
        if not candidate.is_file():
            return None, "The saved snapshot is missing."
        image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None, "The saved snapshot is unreadable."
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None
    except (OSError, RuntimeError, ValueError, cv2.error):
        return None, "Snapshot unavailable."


def _render_live_incident_timeline() -> None:
    """Render a bounded persisted webcam timeline and its snapshot gallery."""
    st.markdown("#### Recent live incident timeline")
    limit = st.selectbox(
        "Recent incidents to show",
        options=[10, 20, 50],
        index=1,
        format_func=lambda count: f"Latest {count}",
        key="live_incident_limit",
    )
    try:
        rows = fetch_recent_alerts_by_source("Browser webcam", limit=limit)
    except Exception:
        logger.exception("Could not load recent browser webcam incidents")
        st.warning("Persisted webcam incidents are temporarily unavailable.")
        return

    current, previous = _partition_live_incidents(
        rows, st.session_state.get("webcam_session_incident_ids", [])
    )
    previews = {}
    for record in rows:
        previews[int(record["id"])] = _load_incident_snapshot(record["snapshot_path"])

    if not rows:
        st.info("No incidents have been persisted from the browser webcam yet.")
    else:
        for title, records in (("This session", current), ("Previously persisted", previous)):
            if not records:
                continue
            st.markdown(f"##### {title}")
            for record in records:
                preview, snapshot_message = previews[int(record["id"])]
                left, middle, right = st.columns([0.65, 3.8, 1.55])
                with left:
                    st.markdown(f"**#{int(record['id'])}**")
                with middle:
                    st.markdown(f"**{record['event_type'] or 'Unknown event'}**")
                    st.caption(_incident_description(record))
                with right:
                    severity = record["severity"] or "Unavailable"
                    color = {"High": "red", "Medium": "orange", "Low": "blue"}.get(severity, "gray")
                    st.badge(severity, color=color)
                    st.caption(format_saved_timestamp_local(record["timestamp"]))
                availability = "Snapshot available" if preview is not None else snapshot_message
                st.caption(f"Source: Browser webcam · {availability}")

    st.markdown("#### Recorded Evidence (Snapshots)")
    if not rows:
        st.info("No persisted webcam incidents are available for snapshot review.")
        return
    columns = st.columns(3)
    for index, record in enumerate(rows):
        preview, snapshot_message = previews[int(record["id"])]
        with columns[index % len(columns)]:
            with st.container(border=True):
                st.markdown(f"**#{int(record['id'])} · {record['event_type'] or 'Unknown event'}**")
                severity = record["severity"] or "Unavailable"
                st.badge(severity, color={"High": "red", "Medium": "orange", "Low": "blue"}.get(severity, "gray"))
                st.caption(format_saved_timestamp_local(record["timestamp"]))
                if preview is None:
                    st.info(snapshot_message)
                else:
                    st.image(preview, width="stretch")


def _persist_webcam_updates(processor: WebcamProcessor) -> None:
    """Persist callback events on the script thread and retain failed writes for retry."""
    from utils.alert_engine import record_incident

    cooldown = st.session_state.alert_cooldown
    pending = st.session_state.webcam_pending_incidents
    pending_types = {event.event_type for _, event in pending}
    session_started_at = st.session_state.webcam_session_started_at
    for update in processor.drain():
        if session_started_at is None or update.processed_at >= session_started_at:
            st.session_state.webcam_last_update = update
            st.session_state.last_detection_metrics = {
                "people": update.result.person_count,
                "crowd": update.result.crowd_density,
                "fps": update.result.fps,
                "source": "Browser webcam",
                "updated_at": time.time(),
            }
        for index, event in enumerate(update.events):
            if event.event_type in pending_types:
                continue
            if cooldown.allow(event.event_type, "Browser webcam"):
                frame = (
                    update.event_frames[index]
                    if index < len(update.event_frames)
                    else update.result.frame
                )
                pending.append((frame, event))
                pending_types.add(event.event_type)

    while pending:
        frame, event = pending[0]
        try:
            incident = record_incident(
                frame,
                event.event_type,
                event.severity,
                event.message,
                "Browser webcam",
                metadata={"box": list(event.box)} if event.box else None,
            )
        except Exception as exc:
            cooldown.forget(event.event_type, "Browser webcam")
            logger.exception("Could not persist browser webcam incident")
            st.session_state.webcam_persist_error = "Incident save failed; retrying."
            st.warning(f"Could not save {event.event_type}. It will be retried automatically.")
            break
        else:
            pending.pop(0)
            cooldown.mark(event.event_type, "Browser webcam")
            st.session_state.webcam_incidents.insert(0, event)
            st.session_state.webcam_incidents = st.session_state.webcam_incidents[:8]
            incident_id = incident.get("id") if isinstance(incident, dict) else None
            if incident_id is not None:
                session_ids = st.session_state.get("webcam_session_incident_ids", [])
                st.session_state.webcam_session_incident_ids = list(
                    dict.fromkeys([int(incident_id), *session_ids])
                )[:200]
                from dashboard.alert_controls import register_persisted_incident

                register_persisted_incident(st.session_state, incident)
            st.session_state.webcam_persist_error = None


def _reset_webcam_session(processor: WebcamProcessor) -> None:
    """Reset temporal tracking without discarding events awaiting persistence."""
    _persist_webcam_updates(processor)
    processor.reset()
    st.session_state.webcam_last_update = None
    st.session_state.webcam_incidents = []
    st.session_state.webcam_session_incident_ids = []
    metrics = st.session_state.last_detection_metrics
    if metrics and metrics.get("source") == "Browser webcam":
        st.session_state.last_detection_metrics = None
    if not st.session_state.webcam_pending_incidents:
        st.session_state.webcam_persist_error = None


@st.fragment(run_every=1.0)
def render_browser_webcam() -> None:
    """Render browser-camera WebRTC with a safe callback-to-Streamlit handoff."""
    init_detection_state()
    if st.session_state.alert_cooldown is None:
        st.session_state.alert_cooldown = AlertCooldown(
            st.session_state.get("alert_cooldown_seconds", ALERT_COOLDOWN_SECONDS)
        )
    if st.session_state.webcam_processor is None:
        st.session_state.webcam_processor = WebcamProcessor(
            st.session_state.detector_manager or DetectionManager(),
            conf_threshold=st.session_state.get("conf_threshold", YOLO_CONFIDENCE_THRESHOLD),
        )
    processor = st.session_state.webcam_processor

    st.info(
        "Allow camera access when prompted. This browser camera requires HTTPS or localhost. "
        "The Start and Stop controls are part of the camera panel."
    )
    if st.button(
        "Reset camera session",
        key="reset_browser_camera",
        disabled=st.session_state.webcam_stream_active,
        help="Stop the stream before resetting its temporal detection state.",
    ):
        _reset_webcam_session(processor)
        st.rerun()

    with st.container(border=True):
        st.markdown("#### Live video")
        st.caption("Browser frames are sent to the app for inference; only incident snapshots are saved.")
        try:
            context = webrtc_streamer(
                key="antadrishti-browser-camera",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": True, "audio": False},
                video_frame_callback=processor.callback,
                async_processing=True,
            )
        except AttributeError as exc:
            if "_session_mgr" not in str(exc):
                logger.exception("WebRTC component failed to initialize")
                st.error("The camera panel could not initialize. Refresh the page and try again.")
                return
            st.error("The camera panel needs an active Streamlit session. Open the running app in a browser.")
            return
    stream_active = bool(context.state.playing)
    if stream_active and not st.session_state.webcam_stream_active:
        st.session_state.webcam_session_started_at = time.monotonic()
        st.session_state.webcam_last_update = None
        metrics = st.session_state.last_detection_metrics
        if metrics and metrics.get("source") == "Browser webcam":
            st.session_state.last_detection_metrics = None
    st.session_state.webcam_stream_active = stream_active
    _persist_webcam_updates(processor)
    from dashboard.alert_controls import render_new_incident_alerts

    render_new_incident_alerts()
    update: WebcamUpdate | None = st.session_state.webcam_last_update
    if update is None:
        if stream_active:
            st.info("Camera streaming · waiting for the first processed frame.")
        else:
            st.info(
                "Camera idle · start the embedded stream and allow camera access. "
                "If access is blocked, check browser permissions and use HTTPS or localhost."
            )
        if not context.state.playing:
            st.button(
                "Analyze a video instead",
                key="webcam_video_fallback",
                on_click=_open_video_analysis,
            )
        return

    if stream_active and time.monotonic() - update.processed_at > 5:
        st.info("Camera streaming · waiting for a fresh processed frame.")
        return

    result = update.result
    st.session_state.last_frame_rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
    if context.state.playing:
        st.success("Streaming · detection active")
    else:
        st.info("Stream stopped · showing the most recent processed frame")
    metrics = st.columns(4)
    metrics[0].metric("People · latest frame", result.person_count)
    metrics[1].metric("Crowd estimate", result.crowd_density)
    metrics[2].metric("Inference FPS", f"{result.fps:.1f}")
    metrics[3].metric("Frame skip", "2 of 3")
    st.caption("Crowd estimate uses detected people and bounding-box occupancy.")
    st.warning(
        "Fall and possible altercation signals are experimental heuristics. "
        "They indicate possible events for human review, not confirmed emergencies."
    )
    if not context.state.playing:
        st.button(
            "Analyze a video instead",
            key="webcam_video_fallback_after_stop",
            on_click=_open_video_analysis,
        )
    _render_live_incident_timeline()
