# AntaDrishti | AI-Powered Public Safety Intelligence

**Transforming video surveillance into a reviewable safety-intelligence workflow.**

AntaDrishti is a computer-vision-based public-safety monitoring prototype that processes webcam frames and recorded videos, detects supported objects using YOLOv8n, evaluates potential safety-related events, and presents incident records through a Streamlit dashboard.

The system combines video processing, object detection, event-processing heuristics, preliminary severity assessment, incident persistence, snapshots, and analytics in one interface.

> **Project status:** Competition prototype. This project supports human review and situational awareness; it is not a validated autonomous safety system.

## Project Links

* **Live Demo:** https://antadrishti.streamlit.app
* **GitHub Repository:** https://github.com/abhishek25001011-boop/AntaDrishti

## Table of Contents

* [The Problem](#the-problem)
* [Our Approach](#our-approach)
* [Key Features](#key-features)
* [System Architecture](#system-architecture)
* [Technology Stack](#technology-stack)
* [Getting Started](#getting-started)
* [Running the Application](#running-the-application)
* [Demo Scenarios](#demo-scenarios)
* [Incident Processing](#incident-processing)
* [Privacy and Data Persistence](#privacy-and-data-persistence)
* [Known Limitations](#known-limitations)
* [Testing](#testing)
* [Future Improvements](#future-improvements)
* [Contributing](#contributing)
* [Responsible Use](#responsible-use)

## The Problem

Public spaces may be monitored through multiple CCTV feeds, requiring operators to divide their attention across different camera views and incident records.

Manually reviewing video can make potentially important events difficult to notice promptly.

AntaDrishti explores how computer vision and event-processing logic can help operators identify, organize, and review potential safety-related events using existing video footage.

## Our Approach

AntaDrishti connects object detection with an incident-review workflow.

Instead of presenting detection boxes alone, the prototype combines object-level detections with event-specific heuristics, preliminary severity labels, incident records, available snapshots, and a centralized dashboard.

The goal is to make potential events easier to inspect while keeping human operators responsible for interpretation and response.

## Key Features

### 1. Dashboard

* View persisted incident totals and recent incident records.
* Review the latest available processed-frame metrics.
* Access an overview of recorded events.

### 2. Live Detection

* Process browser webcam frames using `streamlit-webrtc`.
* Run YOLO inference through a shared detection manager.
* Use a bounded handoff queue for frame processing.
* Save generated events on the Streamlit application thread.
* Display processing status and current-session event information.

Browser camera access requires permission and generally works on `localhost` or an HTTPS origin.

### 3. Video Analysis

* Select bundled sample videos.
* Upload supported MP4, AVI, MOV, or MKV videos.
* Process videos using OpenCV.
* Display processing progress, processed-frame information, and people summaries.
* Review incidents generated during analysis.

Each analysis run processes a maximum of 300 frames.

### 4. Alerts and Incident History

* Browse persisted incident records.
* Filter incidents by event type, severity, and source.
* Inspect incident details and timestamps.
* View snapshots when available.
* Export incident records as CSV.

**Important:** These are application-level incident records and dashboard alerts. The project does not claim that external phone notifications, browser push notifications, email alerts, or voice announcements are available.

### 5. Analytics

* Explore charts derived from persisted SQLite incident records.
* Review recorded event information through the dashboard.

Analytics reflect recorded incidents, not independently validated real-world incident rates.

### 6. Settings

* Inspect model and database availability.
* Review active detection thresholds.

## System Architecture

```text
          Webcam / Recorded Video
                    |
                    v
             OpenCV Processing
                    |
                    v
             YOLOv8n Detection
                    |
                    v
            Detection Manager
                    |
                    v
          Event-Processing Logic
                    |
          +---------+----------+
          |                    |
          v                    v
   Event Assessment      Crowd Assessment
          |                    |
          +---------+----------+
                    |
                    v
        Preliminary Severity Labels
                    |
                    v
           Incident Cooldown
                    |
                    v
         Incident and Snapshot Storage
                    |
             +------+------+
             |             |
             v             v
           SQLite       Snapshots
             |             |
             +------+------+
                    |
                    v
           Streamlit Dashboard
                    |
          +---------+----------+
          |         |          |
          v         v          v
       History   Analytics   Review
```

The diagram represents the logical processing flow. The browser webcam and recorded-video workflows have different input and processing paths.

### Processing Overview

1. **Video input:** Frames come from a browser webcam stream or a selected video file.
2. **Frame processing:** OpenCV handles video decoding and frame preparation.
3. **Object detection:** YOLOv8n identifies supported object classes and returns bounding boxes with confidence scores.
4. **Event processing:** Event-specific logic evaluates detections and relevant visual signals.
5. **Severity assessment:** Generated events receive preliminary severity labels.
6. **Cooldown:** An in-memory, 30-second per-event/source cooldown helps limit repeated incident creation.
7. **Persistence:** Incident records are written to SQLite, and snapshots are saved when available.
8. **Dashboard:** Operators review incidents, snapshots, and analytics.

The cooldown is process-local and resets when the application restarts.

## Technology Stack

| Technology                | Purpose                                         |
| ------------------------- | ----------------------------------------------- |
| Python                    | Application and event-processing logic          |
| Streamlit                 | Interactive dashboard and application interface |
| OpenCV                    | Video decoding and frame processing             |
| Ultralytics YOLOv8n       | Pretrained object detection                     |
| streamlit-webrtc          | Browser webcam frame streaming                  |
| NumPy                     | Numerical and array operations                  |
| pandas                    | Tabular data handling and exports               |
| SQLite                    | Local incident-record persistence               |
| pytest                    | Automated testing                               |
| Git and GitHub            | Version control and source-code hosting         |
| Streamlit Community Cloud | Public demo hosting                             |

The application uses the included `yolov8n.pt` weights, expected at the repository root.

## Getting Started

### Prerequisites

* Python 3.10 or newer
* Git
* A compatible browser for webcam testing
* The included YOLO model weights
* Sufficient memory and processing capacity for video inference

### 1. Clone the Repository

```bash
git clone https://github.com/abhishek25001011-boop/AntaDrishti.git
cd AntaDrishti
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult the official Python documentation for environment activation options rather than changing system-wide execution policies unnecessarily.

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Verify Model Weights

Confirm that the following file exists at the repository root:

```text
yolov8n.pt
```

The application expects these weights to be available locally. Keep the model file consistent with the project's intended configuration.

### 5. Start the Application

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal. The default is usually:

```text
http://localhost:8501
```

## Running the Application

### Browser Webcam Demo

1. Start the application.
2. Open **Live Detection** in the sidebar.
3. Start the embedded webcam stream.
4. Allow camera access when prompted.
5. Wait for processed-frame status and detection metrics.
6. Review events from the current camera session.
7. Stop the stream using the embedded camera controls.

The browser sends webcam frames to the server running the application for processing. Webcam access is subject to browser permissions, origin security requirements, and hardware availability.

### Recorded Video Demo

1. Open **Video Analysis**.
2. Choose **Sample Video** or **Upload Video**.
3. Select a sample clip or a supported video file.
4. Click **Start video analysis**.
5. Monitor progress and processed-frame information.
6. Review any generated incidents and available snapshots.

Uploads are limited to 200 MB in the interface. Uploaded videos are temporarily stored on the server while OpenCV processes them, then the temporary file is removed when processing ends.

Each run processes at most 300 frames. Starting another run begins at the start of the selected video.

## Demo Scenarios

The repository includes these sample clips:

| File                           | Intended Scenario                         |
| ------------------------------ | ----------------------------------------- |
| `sample_videos/demo_valid.mp4` | Basic video-decoding and processing check |
| `sample_videos/bag.mp4`        | Bag-related scenario                      |
| `sample_videos/crowd.mp4`      | Crowd-related scenario                    |
| `sample_videos/fall.mp4`       | Possible-fall scenario                    |
| `sample_videos/fight.mp4`      | Possible-altercation scenario             |

The filenames describe the intended scenarios, not guaranteed detections.

`demo_valid.mp4` is a graphic test clip without people. It checks video decoding and bounded processing; it does not validate person detection or incident detection.

Actual results depend on video content, camera angle, lighting, model outputs, thresholds, and runtime conditions.

## Incident Processing

### Object Detection vs. Event Detection

These are separate stages.

**Object detection** identifies supported objects in individual frames. A detection may include:

* Object class, such as `person` or a recognized bag category.
* Bounding-box coordinates.
* A confidence score.

A confidence score is the model's score for a particular detection. It is not the probability that an incident occurred, nor is it the overall accuracy of the application.

**Event processing** interprets detections using additional rules and visual signals to identify potential events.

For example:

* **Crowd density:** Uses person counts and bounding-box occupancy signals.
* **Possible fall:** Uses a temporal upright-to-horizontal person-box heuristic.
* **Possible altercation:** Uses an experimental optical-flow heuristic around nearby detections.
* **Bag-related event:** Uses detections of supported bag classes.

These mechanisms have different reliability levels and should not be treated as equally validated.

### Incident Records

When an event passes the configured processing and cooldown conditions, the application can create an incident record containing available information such as event type, severity, source, timestamp, and snapshot reference.

Snapshot creation can fail independently of incident persistence.

The system is intended to help operators review potential incidents. It does not independently establish that a crime, medical emergency, or safety violation has occurred.

## Privacy and Data Persistence

### Local Storage

Incident records are stored in:

```text
database/alerts.db
```

Snapshots are written under:

```text
snapshots/
```

On a local run, these files remain on the local machine until removed by the operator.

### Video Processing

* Browser webcam frames are processed by the application server.
* Uploaded videos are temporarily stored server-side during processing.
* The application does not retain a complete video archive through this workflow.
* Snapshots may be retained when incidents are persisted.

### Hosted Deployment Considerations

The project does not configure managed durable storage, authentication, or multi-instance coordination.

Hosted filesystems may be ephemeral. Application restarts or changes in hosting infrastructure can affect locally stored records and snapshots.

Use only footage you are authorized to process. Avoid uploading sensitive footage to a hosted demo unless you understand its access and retention implications.

## Known Limitations

* **Fall detection:** Uses a temporal person-box heuristic. Sitting, bending, occlusion, and detection errors can resemble a fall.
* **Altercation detection:** Uses an experimental optical-flow heuristic. It is not a trained fight-recognition model.
* **Crowd assessment:** Uses frame-relative signals and is not a validated crowd-safety or stampede-prediction system.
* **Bag detection:** Identifies supported bag objects but does not establish whether a bag is unattended or being stolen.
* **Accuracy:** The project does not establish formal benchmark accuracy, precision, or recall for its event categories.
* **Notifications:** External push, email, and phone notifications are not claimed as implemented features.
* **Voice announcements:** Text-to-speech announcements and speaker alerts are not claimed as implemented features.
* **Performance:** Inference speed depends on hardware, video resolution, model execution, and runtime conditions.
* **Video compatibility:** Browser permissions, available codecs, and OpenCV backend support can affect operation.
* **Production readiness:** Authentication, robust persistent storage, privacy controls, and multi-instance coordination are not configured as production services.

## Testing

Install the optional test dependency:

```powershell
python -m pip install pytest
```

Run the test suite:

```powershell
python -m pytest -q
```

Run Python compilation checks:

```powershell
python -m compileall app.py dashboard detectors utils database tests
```

The automated tests cover isolated pipeline, persistence, and UI-state behavior.

Passing automated tests does not establish real-world detection accuracy. Mocked tests do not verify actual browser webcam operation or validate the system under real-world surveillance conditions.

## Future Improvements

Potential next steps include:

* Evaluate event detectors on labelled datasets.
* Measure precision, recall, false-positive rate, and alert latency.
* Improve temporal fall recognition.
* Evaluate action-recognition methods for possible altercations and bag-snatching scenarios.
* Introduce person and object tracking for context-aware event assessment.
* Improve crowd estimation under different camera angles and scene conditions.
* Add configurable notification channels and voice announcements.
* Introduce managed persistent storage and stronger deployment controls.
* Add authentication, access management, privacy safeguards, and audit logs.
* Evaluate multi-camera processing after single-stream reliability is established.

These are proposed improvements, not claims about current functionality.

## Contributing

Contributions that improve reliability, maintainability, documentation, accessibility, testing, and responsible use are welcome.

Suggested contribution workflow:

1. Fork the repository.
2. Create a focused feature branch.
3. Make and test your changes.
4. Describe the changes and any known limitations.
5. Submit a pull request.

For changes to event detection, include representative test cases and document any new assumptions or thresholds.

## Responsible Use

AntaDrishti is a demonstration prototype intended to support human review and situational awareness.

It should not be used as the sole basis for emergency decisions, accusations, disciplinary action, or law-enforcement action. Potential events require appropriate human verification.

Users are responsible for obtaining permission to process footage and following applicable privacy, security, and data-protection requirements.

---

**Built by Team Project Astra**

*AntaDrishti — making video surveillance more actionable through computer vision and reviewable incident workflows.*
