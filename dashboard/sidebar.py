"""Command-center navigation and recorded-video input controls."""

from pathlib import Path

import streamlit as st

from config import (
    APP_DESCRIPTION,
    BASE_DIR,
    MAX_PROCESSING_FRAMES,
    MAX_VIDEO_UPLOAD_MB,
    YOLO_CONFIDENCE_THRESHOLD,
)
from utils.video_utils import list_sample_videos

PAGES = ["Dashboard", "Live Detection", "Video Analysis", "Camera Sources", "Alerts", "Analytics", "Settings"]


def render_sidebar() -> str:
    with st.sidebar:
        logo_path = BASE_DIR / "assets" / "logo.png"
        if logo_path.is_file():
            st.image(str(logo_path), width=132)
        st.markdown("### ANTA DRISHTI")
        st.caption("PROJECT ASTRA")
        st.markdown(f"<div class='ad-muted'>{APP_DESCRIPTION}</div>", unsafe_allow_html=True)
        st.selectbox("Appearance", ["Dark", "Light"], key="theme_mode")
        st.markdown("---")
        st.markdown("<div class='ad-section-label'>Workspace</div>", unsafe_allow_html=True)
        st.radio("Workspace pages", PAGES, key="navigation", label_visibility="collapsed")
        page = st.session_state.navigation

        if page == "Video Analysis":
            st.markdown("---")
            st.markdown("<div class='ad-section-label'>Video input</div>", unsafe_allow_html=True)
            source = st.radio(
                "Analysis source",
                ["Webcam", "Upload Video", "Sample Video"],
                index=2,
                key="input_source",
            )
            uploaded_file = None
            if source == "Upload Video":
                uploaded_file = st.file_uploader(
                    "Choose a video",
                    type=["mp4", "mov", "avi", "mkv"],
                    key="uploaded_video",
                    max_upload_size=MAX_VIDEO_UPLOAD_MB,
                )
            if source == "Webcam":
                st.caption("Uses the camera attached to the machine running Streamlit, not this browser camera.")

            sample_video_paths = list_sample_videos()
            default_sample_index = next(
                (i for i, path in enumerate(sample_video_paths) if Path(path).name == "demo_valid.mp4"), 0
            )
            sample_video_path = st.selectbox(
                "Sample clip",
                options=sample_video_paths or ["No valid sample videos found"],
                index=default_sample_index if sample_video_paths else 0,
                key="sample_video_path",
                disabled=not sample_video_paths,
                format_func=lambda path: Path(path).name.replace("_", " ") if sample_video_paths else path,
            )
            st.number_input(
                "YOLO confidence",
                min_value=0.10,
                max_value=0.90,
                value=YOLO_CONFIDENCE_THRESHOLD,
                step=0.05,
                key="conf_threshold",
                help="Minimum confidence threshold for object detections.",
            )
            st.caption(f"Bounded run · up to {MAX_PROCESSING_FRAMES} frames")

            if st.button("Start video analysis", width="stretch", type="primary", key="start_detection"):
                if source == "Upload Video" and uploaded_file is None:
                    st.warning("Choose a video file before starting.")
                elif source == "Sample Video" and not sample_video_paths:
                    st.warning("No nonempty sample videos are available.")
                else:
                    st.session_state.detection_running = True
                    st.session_state.selected_source = source
                    st.session_state.selected_uploaded_video = uploaded_file
                    st.session_state.selected_sample_video_path = (
                        sample_video_path if sample_video_paths else None
                    )

        st.markdown("---")
        st.caption("Fall and possible altercation signals are experimental heuristics.")
    return page
