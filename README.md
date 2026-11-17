# AI-Powered Aim & Mechanics Telemetry System

This project is an **AI-powered Aim & Mechanics Telemetry System**. Its purpose is to analyse gameplay footage or live screen captures, track targets, measure aiming behaviour (reaction latency, crosshair placement drift, overshoot), and display real-time performance metrics in a decoupled dashboard.

---

## ⚡ Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# Or directly:
pip install mss opencv-python numpy requests pillow
```

*(Optional: for custom neural models, install `onnxruntime` or `onnxruntime-gpu`)*

### 2. Run the Dashboard

```bash
python3 telemetry_app.py
```

- Choose **Live Screen (mss)** for real-time capture from your primary monitor.
- Or choose **VOD File**, click **Select VOD** to load an `.mp4`/`.mkv`/`.avi` recording, and click **Start Capture**.

---

## 📁 Project Structure

```text
aim_analyser/
├── vision_engine.py      # Core vision & mechanics pipeline (screen/VOD capture, tracking, analytics)
├── telemetry_app.py      # Responsive Tkinter telemetry dashboard & API benchmark sync
├── requirements.txt      # Python dependencies
├── README.md             # Project overview & quickstart
└── ARCHITECTURE.md       # Comprehensive architecture deep-dive & roadmap
```

---

## 🚀 Key Modules

1. **`vision_engine.py`** — The brain. Isolates computer-vision and telemetry logic:
   - High-FPS screen capture using `mss` (plus OpenCV VOD fallback).
   - Swappable detector interface: `BaseMechanicTracker`.
   - Out-of-the-box `ColorTargetTracker` (HSV + centroid tracking for Aim Lab / Kovaak's / enemy outlines).
   - `ONNXCustomTracker` interface for custom models (RT-DETR, YOLO, RF-DETR).
   - `MechanicsAnalyzer` for reaction latency, vertical head-drift, and flick overshoot computation.

2. **`telemetry_app.py`** — The interface:
   - Non-blocking 60Hz Tkinter GUI.
   - Background worker thread for capture and inference.
   - Thread-safe UI updates via `root.after()`.
   - Asynchronous player API baseline comparison & live diagnostics feed.

For complete architectural details, algorithms, limitations, and future roadmap, refer to [ARCHITECTURE.md](file:///home/vedant-pai/Documents/aim_analyser/ARCHITECTURE.md).
