# AntaDrishti

AntaDrishti is a Streamlit public-safety monitoring prototype. It processes a webcam, uploaded video, or a nonempty sample video with the included YOLOv8n COCO weights. Person and bag detections feed frame-relative crowd estimates and experimental temporal event heuristics. Incidents are saved with a timestamped snapshot and SQLite history.

## Current detection scope

- **People and bags:** YOLOv8n object detection using the root-level `yolov8n.pt` weights.
- **Crowd density:** estimated from detected person count and the fraction of frame covered by person boxes.
- **Possible fall:** a temporal upright-to-horizontal person-box heuristic. It can confuse sitting, bending, or occlusion with a fall.
- **Possible physical altercation:** an experimental heuristic that looks for repeated high optical flow around nearby detected people. There is no trained fight detector, and this can flag unrelated activity.
- **Severity:** incidents are stored as Low, Medium, or High.

These are prototype signals for review, not validated safety systems. The checked-in `sample_videos/demo_valid.mp4` is a graphic test clip without people; it is useful for validating video decoding but cannot demonstrate person or event detection. Add suitable footage or use a camera/upload to exercise those detections.

## Requirements

- Python 3.10 or newer
- Streamlit 1.51 or newer
- The included root-level `yolov8n.pt` model file
- A webcam for camera input, or a readable video file for upload/sample processing

## Install and run (Windows PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Choose **Webcam**, **Upload Video**, or a listed nonempty **Sample Video** from the sidebar. Select a confidence threshold and press **Start Detection**. Each run processes up to 300 frames, shows the latest processed frame and progress, then stops automatically and returns control. For long videos, a new run starts from the beginning; it does not resume.

Uploads accept MP4, AVI, MOV, and MKV where the installed OpenCV video backend can decode them. The app copies the upload to a temporary file while processing and removes it afterward. Webcam mode opens the local camera attached to the machine running Streamlit; a remotely hosted Streamlit app cannot access the viewer's camera. Camera runs are also bounded to 300 frames.

The sample selector lists only nonempty files. The supplied `demo_valid.mp4` is a graphic clip with no real people, so it verifies decoding and bounded processing only. Use real video with visible people to evaluate YOLO and the event heuristics.

The sidebar navigation contains **Dashboard**, **Live Detection**, **Video Analysis**, **Alerts**, **Analytics**, and **Settings**. Alert history supports event/severity/source filters, incident details, snapshot viewing, and CSV export. Analytics are computed from stored SQLite incidents only. Settings shows the model/database status and active heuristic thresholds; confidence and the session alert cooldown are connected controls.

## Data locations

- `database/alerts.db`: incident records. Existing `alerts` tables are upgraded in place while legacy columns are retained.
- `snapshots/`: timestamped JPEG incident images.
- `sample_videos/`: selectable local video inputs; empty files are ignored.

## Known limits

- The bundled sample clip has no real people, so person detection must be tried with real footage.
- Camera availability and codec support depend on the machine running Streamlit.
- Fight and fall heuristics require person detections and are not machine-learned event classifiers.
- Repeated event/source pairs are suppressed for 30 seconds by an in-memory cooldown. Restarting the app clears that cooldown.

## Internal tests

Run the isolated unit tests with:

```powershell
python -m unittest discover -s tests -v
```

The tests use synthetic frames only for algorithm unit checks and a temporary SQLite database; they do not create production incidents.
