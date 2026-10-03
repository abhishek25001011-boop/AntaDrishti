"""Bounded, user-started processing for the selected video or camera source."""

import cv2
import streamlit as st

from config import ALERT_COOLDOWN_SECONDS, CAMERA_SOURCE, MAX_PROCESSING_FRAMES, YOLO_CONFIDENCE_THRESHOLD
from database.database import count_alerts
from detectors.detection_manager import DetectionManager
from utils.alert_cooldown import AlertCooldown
from utils.event_detection import EventDetector
from utils.video_utils import open_video_capture, release_capture
from utils.video_pipeline import process_capture


def init_detection_state() -> None:
    defaults = {
        "detection_running": False,
        "video_capture": None,
        "capture_source": None,
        "detector_manager": None,
        "alert_cooldown": None,
        "selected_source": "Webcam",
        "last_summary": None,
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
        if st.session_state.last_summary:
            summary = st.session_state.last_summary
            status_area.info(
                f"Last run processed {summary.frames_processed} frame(s). "
                f"Incidents saved: {len(summary.incidents)}. Press Start Detection to run again."
            )
        else:
            video_area.info("Choose a source in the sidebar and press Start Detection.")
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

        def update_frame(index, limit, total, result, incidents):
            target = min(total, limit) if total else limit
            progress_area.progress(min(1.0, index / max(1, target)), text=f"Processing frame {index} of {target} · source: {event_source}")
            rgb_frame = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
            st.session_state.last_frame_rgb = rgb_frame
            video_area.image(rgb_frame, channels="RGB", width="stretch")
            status_area.write(
                f"Detection Active · Source: {event_source} | Frame: {index} | FPS: {result.fps:.1f} | "
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
    except Exception as exc:
        st.session_state.last_processing_error = str(exc)
        status_area.error(f"Processing failed: {exc}")
    finally:
        release_capture(capture)
        st.session_state.video_capture = None
        st.session_state.capture_source = None
        st.session_state.detection_running = False
