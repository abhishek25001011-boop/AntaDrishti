"""Sidebar components for the Streamlit dashboard."""

from pathlib import Path

import streamlit as st

from config import APP_DESCRIPTION, APP_NAME, SAMPLE_VIDEO_DIR
from utils.video_utils import release_capture


def render_sidebar() -> None:
    """Render the application sidebar and detection controls."""
    with st.sidebar:
        st.header(APP_NAME)
        st.write(APP_DESCRIPTION)

        source = st.radio("Source", ["Webcam", "Upload Video", "Sample Video"], index=2)
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"], key="uploaded_video")
        sample_video_paths = [str(path) for path in sorted(SAMPLE_VIDEO_DIR.glob("*")) if path.is_file()] if SAMPLE_VIDEO_DIR.exists() else []
        default_sample_index = 0
        for index, path in enumerate(sample_video_paths):
            if path.endswith("demo_valid.mp4"):
                default_sample_index = index
                break
        sample_video_path = st.selectbox(
            "Sample Video",
            options=sample_video_paths if sample_video_paths else ["No sample videos found"],
            index=default_sample_index if sample_video_paths else 0,
            key="sample_video_path",
            disabled=not sample_video_paths,
        )
        st.number_input("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05, key="conf_threshold")

        if st.button("Start Detection", use_container_width=True, key="start_detection"):
            if source == "Upload Video" and uploaded_file is None:
                st.warning("Please upload a video file before starting detection.")
            elif source == "Sample Video" and (not sample_video_paths or sample_video_path == "No sample videos found"):
                st.warning("Please add a sample video before starting detection.")
            else:
                st.session_state.detection_running = True
                st.session_state.selected_source = source
                st.session_state.selected_uploaded_video = uploaded_file
                st.session_state.selected_sample_video_path = sample_video_path if sample_video_paths else None

        if st.button("Stop Detection", use_container_width=True, key="stop_detection"):
            st.session_state.detection_running = False
            release_capture(st.session_state.get("video_capture"))
            st.session_state.video_capture = None
            st.session_state.capture_source = None
            st.session_state.fall_timer = None
            st.session_state.fall_active = False
            st.session_state.fall_alert_created = False
            st.session_state.fall_alert_snapshot_id = None

        st.markdown("---")
        st.caption("Use Start/Stop and video source controls for live detection.")
