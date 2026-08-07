"""Live detection rendering and camera handling for the dashboard."""

import cv2
import streamlit as st
import time

from database.database import count_alerts
from detectors.detection_manager import DetectionManager
from utils.fall_detector import annotate_fall, create_fall_alert, find_fallen_person
from utils.video_utils import open_video_capture, release_capture


def init_detection_state() -> None:
    """Ensure Streamlit session state keys exist for detection."""
    defaults = {
        "detection_running": False,
        "video_capture": None,
        "capture_source": None,
        "detector_manager": None,
        "selected_source": "Webcam",
        "last_fps": 0.0,
        "person_count": 0,
        "bag_count": 0,
        "capture_error": None,
        "fall_timer": None,
        "fall_active": False,
        "fall_alert_created": False,
        "fall_alert_snapshot_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_live_detection() -> None:
    """Render the live detection feed and update metrics."""
    init_detection_state()

    source = st.session_state.get("selected_source", "Webcam")
    uploaded_file = st.session_state.get("selected_uploaded_video")
    sample_video_path = st.session_state.get("selected_sample_video_path")
    conf_threshold = st.session_state.get("conf_threshold", 0.35)

    if source == "Sample Video":
        preferred_sample = st.session_state.get("selected_sample_video_path")
        if not preferred_sample or not str(preferred_sample).endswith("demo_valid.mp4"):
            sample_video_path = None

    if st.session_state.detector_manager is None:
        st.session_state.detector_manager = DetectionManager()

    source_key = (source, getattr(uploaded_file, "name", None), sample_video_path)
    if st.session_state.detection_running and (
        st.session_state.video_capture is None or st.session_state.capture_source != source_key
    ):
        release_capture(st.session_state.get("video_capture"))
        capture = open_video_capture(source, uploaded_file, sample_video_path)
        st.session_state.video_capture = capture
        st.session_state.capture_source = source_key
        st.session_state.capture_error = None if capture is not None and capture.isOpened() else "Could not open any available video source."

    video_area = st.empty()
    status_area = st.empty()

    if not st.session_state.detection_running:
        video_area.info("Detection is stopped. Press Start Detection to begin.")
        return

    capture = st.session_state.video_capture
    if capture is None or not capture.isOpened():
        reason = st.session_state.get("capture_error") or "Could not open any available video source."
        status_area.error(reason)
        st.session_state.detection_running = False
        return

    ret, frame = capture.read()
    if not ret:
        status_area.warning("Video source ended or frame could not be read.")
        st.session_state.detection_running = False
        release_capture(capture)
        st.session_state.video_capture = None
        st.session_state.capture_source = None
        return

    detection_result = st.session_state.detector_manager.process_frame(frame, conf_threshold)
    output_frame = detection_result.frame
    person_count = detection_result.person_count
    bag_count = detection_result.bag_count
    st.session_state.last_fps = round(detection_result.fps, 2)
    st.session_state.person_count = person_count
    st.session_state.bag_count = bag_count

    fallen_persons = find_fallen_person(detection_result.detections)
    current_time = time.time()

    if fallen_persons:
        if st.session_state.fall_timer is None:
            st.session_state.fall_timer = current_time

        if current_time - st.session_state.fall_timer >= 3.0:
            st.session_state.fall_active = True
            if not st.session_state.fall_alert_created:
                first_box = fallen_persons[0]["box"]
                output_frame = annotate_fall(
                    output_frame, first_box, "MEDICAL EMERGENCY DETECTED"
                )
                st.session_state.fall_alert_snapshot_id = create_fall_alert(
                    frame, "MEDICAL EMERGENCY DETECTED"
                )
                st.session_state.fall_alert_created = True
            else:
                for person in fallen_persons:
                    output_frame = annotate_fall(
                        output_frame, person["box"], "MEDICAL EMERGENCY DETECTED"
                    )
    else:
        st.session_state.fall_timer = None
        st.session_state.fall_active = False
        st.session_state.fall_alert_created = False
        st.session_state.fall_alert_snapshot_id = None

    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    video_area.image(output_frame, channels="RGB", use_column_width=True)
    status_area.write(
        f"FPS: {st.session_state.last_fps} | Persons: {person_count} | Bags: {bag_count} | Crowd: {detection_result.crowd_density}"
    )
    st.caption(
        f"Active Alerts: {count_alerts(status='Active')} | Critical Alerts: {count_alerts(severity='Critical')}"
    )

    if st.session_state.detection_running:
        time.sleep(0.01)
        st.rerun()
