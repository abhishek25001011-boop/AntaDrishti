# AntaDrishti

AntaDrishti is a Streamlit public-safety monitoring prototype. It processes browser webcam frames or recorded video, detects supported objects with YOLOv8n, assesses crowd density, and saves generated incidents to local SQLite history with snapshots when available.

## Problem and approach

Public-space monitoring can require an operator to review multiple camera feeds and incident records. This prototype brings frame processing, event signals, and incident review into one local dashboard. It is a demonstration and review aid, not a validated or autonomous safety system.

## Features and architecture

- **Dashboard:** persisted incident totals, recent incidents, and latest processed-frame metrics.
- **Live Detection:** browser camera through WebRTC, frame callback, cached YOLO inference, and a bounded handoff queue. Events are saved on the Streamlit thread.
- **Video Analysis:** select an included sample clip or upload an MP4, AVI, MOV, or MKV file. Processing is bounded to 300 frames per run and shows progress and results.
- **Alerts / History:** SQLite-backed incident list, event/severity/source filters, incident details, snapshot viewing, and CSV export.
- **Analytics:** charts derived from persisted SQLite incidents.
- **Settings:** model/database availability and active thresholds.

The shared `DetectionManager` loads `yolov8n.pt` lazily and reuses it. YOLOv8n COCO detections support people and recognized bag classes. Crowd assessment uses person counts and bounding-box occupancy. Incident events pass through event detection and an in-memory 30-second per-event/source cooldown before persistence.

## Technology

- Python 3.10 or newer
- Streamlit and `streamlit-webrtc`
- OpenCV headless, Ultralytics, NumPy, and pandas
- SQLite
- Included `yolov8n.pt` weights

Dependencies are listed in `requirements.txt`. This repository does not include a deployment manifest or managed persistent database configuration.

## Install and run locally (Windows PowerShell)

From the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. The model weights are expected at the repository root as `yolov8n.pt`; the app does not need to download them when that file is present.

## Browser webcam demo

1. Open **Live Detection** in the sidebar.
2. Start the embedded camera stream and allow camera access in the browser prompt.
3. Use `localhost` or HTTPS. Browsers generally block webcam access on an insecure remote HTTP origin.
4. Wait for the processed-frame status and metrics. Stop the stream using the embedded camera controls. Recent events from the current camera session appear on this page.

The browser sends frames to the running Streamlit app for inference. The browser webcam is separate from the local OpenCV camera input available in Video Analysis; it is not treated as an uploaded video.

## Sample-video and upload demo

1. Open **Video Analysis**.
2. Choose **Sample Video** and a listed clip, or choose **Upload Video** and select a supported file.
3. Press **Start video analysis**. The page displays processing progress, processed-frame and people summary, and any persisted incidents.

`sample_videos/demo_valid.mp4` is a graphic test clip without people. It verifies video decoding and bounded processing, not person or incident detection. Other bundled clips are `sample_videos/bag.mp4`, `sample_videos/crowd.mp4`, `sample_videos/fall.mp4`, and `sample_videos/fight.mp4`; their filenames describe the intended demo scenario and do not establish detection accuracy. Inference depends on the actual frames, model, and runtime.

Uploads are limited to 200 MB in the interface. A temporary file is used while OpenCV processes an upload and is removed when processing ends. A run handles at most 300 frames; starting another run begins at the start of the selected video.

## Incidents, privacy, and persistence

- SQLite incident records are stored at `database/alerts.db`.
- Incident snapshots are written under `snapshots/`; snapshot failure can leave the incident record without an image.
- Browser webcam frames and uploaded video are processed by the server running this app. Video uploads use a temporary server-side file. Frames are not retained as a video archive; event snapshots are saved when incidents are persisted.
- On a local run, database and snapshot files remain on that machine until removed by its operator. On hosted deployments, they remain on the host filesystem; ephemeral or scaled hosts may lose or split local files. This project does not configure durable shared storage, authentication, or multi-instance coordination.
- The cooldown is in memory and resets when the process restarts. SQLite persistence alone does not make a hosted filesystem durable.

Use footage you are authorized to process. Avoid sensitive footage in a hosted demo unless the host's access and retention behavior is understood.

## Known limitations

- Possible falls are based on a temporal upright-to-horizontal person-box heuristic. Sitting, bending, occlusion, and detector errors can resemble a fall.
- Possible altercations use an experimental optical-flow heuristic around nearby detections. This is not a trained fight detector and can flag unrelated motion.
- Crowd density is a frame-relative estimate, not a validated crowd-safety measure. Object detection and event signals have not been assigned accuracy claims by this project.
- Camera permission, camera availability, inference speed, and supported video codecs depend on the browser, host, hardware, and installed OpenCV backend.
- The application is a competition prototype. It has no built-in user authentication or configured production deployment/persistence service.

## Tests

From the repository root, with project dependencies installed:

```powershell
python -m pip install pytest
python -m pytest -q
python -m compileall app.py dashboard detectors utils database tests
```

`pytest` is optional test tooling and is not an application dependency in `requirements.txt`.
Automated tests cover isolated pipeline, persistence, and UI-state behavior. Mocked tests do not verify a real browser webcam session or real-world detector accuracy.
