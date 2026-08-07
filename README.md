# ANTAHDRISHTI

## Project Overview
AntaDrishti is a beginner-friendly, hackathon-ready prototype for public safety monitoring. The goal is to showcase a modular Python + Streamlit dashboard for detecting and logging safety incidents such as falls, violence, crowding, and unattended bags.

## Features
- Clean Streamlit dashboard
- Sidebar controls and project overview
- SQLite-backed alert history
- Modular detector placeholders
- Simple snapshot and severity helpers

## Folder Structure
- app.py: Main dashboard entry point
- config.py: Global settings
- database/: SQLite database logic
- detectors/: Detection module placeholders
- dashboard/: UI components
- utils/: Helper utilities
- models/: Model files folder
- outputs/ and snapshots/: Output storage

## Installation
1. Create and activate a virtual environment.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py

## Run Command
streamlit run app.py

## Future Scope
- Replace placeholder detectors with YOLOv8 inference
- Add real-time alerts and notifications
- Connect a live camera feed
- Add Gemini API integration for summarization
