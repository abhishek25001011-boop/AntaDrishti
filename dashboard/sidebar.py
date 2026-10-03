"""Command-center navigation and source controls."""

import streamlit as st

from config import (
    APP_DESCRIPTION,
    APP_NAME,
    BASE_DIR,
    MAX_PROCESSING_FRAMES,
    YOLO_CONFIDENCE_THRESHOLD,
)
from utils.video_utils import list_sample_videos

PAGES = ["Dashboard", "Live Detection", "Video Analysis", "Alerts", "Analytics", "Settings"]


def render_sidebar() -> str:
    with st.sidebar:
        logo_path = BASE_DIR / "assets" / "logo.png"
        if logo_path.is_file():
            st.image(str(logo_path), width=180)
        st.markdown("### ◈ ANTA**DRISHTI**")
        st.caption("PROJECT ASTRA · SAFETY INTELLIGENCE")
        st.write(APP_DESCRIPTION)
        st.radio("Navigation", PAGES, key="navigation", label_visibility="collapsed")
        page = st.session_state.navigation

        if page in {"Live Detection", "Video Analysis"}:
            st.markdown("---")
            st.markdown("#### Input Controls")
            source = st.radio(
                "Source", ["Webcam", "Upload Video", "Sample Video"],
                index=2, key="input_source",
            )
            uploaded_file = st.file_uploader(
                "Upload Video", type=["mp4", "mov", "avi", "mkv"], key="uploaded_video"
            )
            if source == "Webcam":
                st.caption("Uses the camera attached to the machine running Streamlit.")
            sample_video_paths = list_sample_videos()
            default_sample_index = next(
                (i for i, path in enumerate(sample_video_paths) if path.endswith("demo_valid.mp4")), 0
            )
            sample_video_path = st.selectbox(
                "Sample Video",
                options=sample_video_paths or ["No valid sample videos found"],
                index=default_sample_index if sample_video_paths else 0,
                key="sample_video_path",
                disabled=not sample_video_paths,
            )
            st.number_input(
                "YOLO confidence", min_value=0.10, max_value=0.90,
                value=YOLO_CONFIDENCE_THRESHOLD, step=0.05, key="conf_threshold",
            )
            st.caption(f"Run limit: {MAX_PROCESSING_FRAMES} frames")

            if st.button("Start Detection", width="stretch", type="primary", key="start_detection"):
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
        st.caption("Fall and altercation alerts are experimental heuristics.")
    return page
