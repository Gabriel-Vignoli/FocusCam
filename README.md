# PhoneGuard

A real-time webcam monitoring tool that uses YOLOv8 object detection to catch you 
reaching for your phone — and calls you out for it.

## What it does
- Continuously analyzes your webcam feed using a YOLOv8 object detection model
- Detects when a cell phone appears in frame
- Displays a bold on-screen warning ("LARGA O CELULAR!") with a bounding box around 
  the detected phone
- Plays an alert sound and sends a desktop notification
- Includes a stop button to silence the alert mid-session
- Also outlines detected people and chairs for visual context (non-triggering)

## Why
Built as a lightweight self-discipline tool to reduce phone distractions during 
study/work sessions in front of the computer.

## Tech stack
- Python
- OpenCV — camera capture and rendering
- Ultralytics YOLOv8 — object detection
- Pygame — audio alerts
- Plyer — desktop notifications

## Setup
\```bash
pip install opencv-python ultralytics plyer pygame
python monitor.py
\```

Place an `alert.ogg` (or `.mp3`) file in the project folder for the sound alert.

## Notes
- Requires a webcam
- Tested on Windows (uses DirectShow backend via `cv2.CAP_DSHOW`)
- Press `q` to quit