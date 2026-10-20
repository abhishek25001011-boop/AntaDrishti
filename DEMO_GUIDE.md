# AntaDrishti competition demo

## Before the demo

- Run from the repository root and confirm `yolov8n.pt` exists there.
- Have a browser camera available and use `localhost` or an HTTPS address.
- Keep `sample_videos/bag.mp4` ready as the incident-pipeline fallback. `sample_videos/demo_valid.mp4` is a no-people decoding check, not an incident demo.

## Exact startup command

In Windows PowerShell, from the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If the project environment has not been created yet, follow the installation steps in `README.md` first. Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Three-minute sequence

| Time | Show |
| --- | --- |
| 0:00–0:20 | **Dashboard:** explain the persisted incident totals and point out that the frame and metrics refer to the latest processed input. |
| 0:20–1:05 | **Live Detection:** open the page, start the embedded browser camera, allow the browser permission prompt, and wait for a fresh processed-frame status and metrics. |
| 1:05–1:35 | **One supported incident:** if a bag is clearly visible and Bag Detection is actually reported, show the live event. If not, use the recorded `bag.mp4` scenario in Video Analysis as the honest fallback; inference is not guaranteed. Do not stage a fall or altercation. |
| 1:35–2:05 | **Alerts/History:** show the saved event, its source (`Browser webcam` or the video filename), timestamp, severity, and snapshot when available. |
| 2:05–2:30 | **Analytics:** show charts and totals derived from persisted SQLite incidents. |
| 2:30–3:00 | **Sample-video fallback:** run a listed sample clip in Video Analysis. Use `bag.mp4` for an intended incident scenario; use `demo_valid.mp4` only to show decoding and bounded completion when inference is unavailable. |

The browser webcam is under **Live Detection**. **Video Analysis** processes a selected uploaded or sample video and does not present webcam frames as recorded-video results.

## If webcam permission or inference fails

1. Confirm the browser is on `localhost` or HTTPS. If no prompt appeared, check that site's camera permission in browser settings, allow access, then reload the page and start the embedded stream again.
2. If the page says it is waiting for a processed frame, check **Settings** for model-file availability and confirm the camera is not blocked or in use by another application.
3. Continue with **Video Analysis → Sample Video → `bag.mp4` → Start video analysis**. Show the actual completion and any actual event. If none is reported, say so; do not present the clip as a successful detection.
4. If only `demo_valid.mp4` is used, explain that it contains no people and demonstrates video decoding only.

## Limitations to state

Person and supported bag classes use the included YOLOv8n COCO weights. Crowd assessment uses counts and box occupancy. Fall and possible altercation signals are experimental heuristics, not trained event classifiers or confirmed incidents. Results depend on the input and runtime; this demo makes no accuracy claim. Incidents and snapshots are stored on the app host in SQLite and the `snapshots/` directory; hosted local storage may not be durable.

## Operator appearance and alert controls

- **Appearance** in the sidebar switches the app surface between Dark and Light. The choice is held in Streamlit session state and lasts for that session; it is not a user-account preference and may reset after a browser refresh or a new session.
- **Settings → Alert preferences** contains independent visual, browser notification, sound, and voice controls. These choices are session-only. Browser notification permission plus local playback-unlock, mute, and incident-ID deduplication markers are managed by that browser for this app origin until its site storage is cleared; these markers do not contain incident descriptions or snapshots.
- Browser notifications require an explicit click on **Enable browser notifications**, and a supported secure browser context such as HTTPS or localhost. They are delivered only while the app page is open; this build has no service worker or background push service.
- Sound and spoken alerts require the explicit **Enable alert sound** / **Enable voice alerts** action in the browser controls. Speech uses the browser's built-in speech synthesis and available voices; English and Hinglish are selectable. Browser policy, available voices, and autoplay restrictions can prevent playback.
- A camera is shown as connected only while the browser WebRTC stream reports active. The local camera is available only for bounded, on-demand Video Analysis on the machine hosting Streamlit. RTSP/IP ingestion and simultaneous multi-camera inference are not supported.
- High-severity visual alerts are based on newly persisted incidents in the current Streamlit session and can be acknowledged without deleting the incident. Possible fall and altercation signals remain heuristics requiring human review.

Test notifications, audio, voice, both appearance modes, and camera permission in the actual demo browser beforehand. Automated tests and Streamlit AppTest do not verify those browser APIs or a physical camera.
