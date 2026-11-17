# Aim & Mechanics Telemetry System — Architecture & Engineering Guide

This project is an **AI-powered Aim & Mechanics Telemetry System**. Its purpose is to analyse your gameplay footage or live screen, track targets, measure your aiming behaviour, and display performance metrics in a dashboard.

The architecture separates the application into two major components:

1. **`vision_engine.py`** – The brain. Captures frames, detects targets, tracks their movement, and calculates aiming metrics.
2. **`telemetry_app.py`** – The interface. Displays the video feed, performance statistics, diagnostics, and external benchmarks.

---

# 1. Overall Architecture

Understanding how components communicate:

```text
                 USER
                  |
                  v
          TELEMETRY_APP.PY
          (Tkinter Dashboard)
                  |
                  | Starts capture worker
                  v
           VISION_ENGINE.PY
                  |
          +-------+-------+
          |               |
          v               v
     Screen Capture    VOD Capture
       (mss)           (OpenCV)
          |               |
          +-------+-------+
                  |
                  v
          DETECTOR INTERFACE
          BaseMechanicTracker
                  |
          +-------+-------+
          |               |
          v               v
    ColorTargetTracker  ONNXCustomTracker
          |               |
          +-------+-------+
                  |
                  v
          MechanicsAnalyzer
                  |
                  v
           FrameTelemetry
                  |
                  v
          TELEMETRY_APP.PY
                  |
          +-------+--------+
          |                |
          v                v
     Video Display    Metrics Dashboard
                          |
                          v
                   Diagnostics/API
```

The detector and dashboard are strictly decoupled. You can replace the detection algorithm without rewriting the dashboard.

---

# 2. `vision_engine.py` – The Computer Vision Engine

This file isolates all computer-vision logic behind five major responsibilities:
1. Represent detected targets.
2. Capture gameplay footage.
3. Detect and track targets.
4. Calculate aiming mechanics.
5. Return telemetry to the dashboard.

---

## A. `TrackedEntity` – Represents a Detected Target

```python
@dataclass
class TrackedEntity:
    track_id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
```

Whenever the detector identifies a target, it instantiates a `TrackedEntity`.

For example:
```python
TrackedEntity(
    track_id=1,
    label="enemy",
    confidence=0.95,
    bbox=(400, 200, 480, 300),
    center=(440, 250)
)
```

| Attribute | Meaning |
|---|---|
| `track_id` | Unique identifier assigned to the target |
| `label` | Target classification (e.g. enemy, target, head) |
| `confidence` | Model confidence score (0.0 to 1.0) |
| `bbox` | Bounding box coordinates `(x1, y1, x2, y2)` |
| `center` | Centre coordinates `(cx, cy)` |

### Bounding Box Geometry:
```text
       (400, 200)
           +----------------+
           |                |
           |     ENEMY      |
           |                |
           +----------------+
                         (480, 300)
```
These coordinates allow the system to calculate crosshair distance, head alignment, and acquisition times.

---

## B. `FrameTelemetry` – Stores Performance Metrics

```python
@dataclass
class FrameTelemetry:
    fps: float = 0.0
    active_targets: int = 0
    crosshair_drift_px: float = 0.0
    aim_offset_px: float = 0.0
    avg_reaction_ms: float = 0.0
    overshoot_count: int = 0
    annotated_frame: Optional[np.ndarray] = None
    event_log: Optional[str] = None
```

This data container is transmitted from the vision worker to the UI thread:

| Metric | Meaning |
|---|---|
| `fps` | Frame capture & processing rate |
| `active_targets` | Number of currently tracked targets |
| `crosshair_drift_px` | Distance from calibrated head-level line |
| `aim_offset_px` | Distance from screen centre (crosshair) to nearest target |
| `avg_reaction_ms` | Time between target detection and crosshair lock |
| `overshoot_count` | Number of flick overshoot events detected |
| `annotated_frame` | Processed BGR image with bounding boxes & HUD overlays |
| `event_log` | Event string (e.g. `"Target #1 acquired in 235 ms"`) |

---

## C. `BaseMechanicTracker` – The Swappable Detector Interface

```python
class BaseMechanicTracker(ABC):

    @abstractmethod
    def detect_and_track(self, frame: np.ndarray) -> List[TrackedEntity]:
        pass
```

This abstract base class acts as an architectural contract. Any detector (traditional CV, YOLO, RT-DETR, RF-DETR) implements `detect_and_track(frame)` and returns `List[TrackedEntity]`.

```text
                 BaseMechanicTracker
                         |
            +------------+------------+
            |            |            |
            v            v            v
       ColorTracker   YOLOTracker   RTDETRTracker
```

Because `MechanicsAnalyzer` only interacts with `BaseMechanicTracker`, swapping detector backends requires zero changes to the telemetry or dashboard logic.

---

## D. `ColorTargetTracker` – Traditional Computer Vision

Ships out of the box with zero external model dependencies. It leverages HSV colour segmentation:

1. **Color Space Conversion**:
   ```python
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   ```
2. **Thresholding & Morphology**:
   ```python
   mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
   mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
   ```
3. **Contour Extraction & Bounding Boxes**:
   Contours filtered by `min_area` generate bounding boxes and centroid coordinates.
4. **Centroid Matching**:
   A greedy nearest-neighbour distance check (within 85 px) preserves IDs across consecutive frames.

---

## E. `MechanicsAnalyzer` – Mechanics Metrics Computation

### 1. FPS Measurement
Calculated from wall-clock delta:
```python
dt = max(now - self._prev_time, 1e-6)
fps = 1.0 / dt
```

### 2. Crosshair Position
Assuming standard centred crosshair:
```python
cx_screen = w // 2
cy_screen = h // 2
```

### 3. Head-Level Drift
Calibrated horizon at `head_level_ratio` (default `0.48`):
```python
head_y = int(h * self.head_level_ratio)
vertical_drift = abs(ty - head_y)
```

### 4. Aim Offset
Euclidean distance from crosshair to the nearest target:
$$D = \sqrt{(x_{\text{target}} - x_{\text{crosshair}})^2 + (y_{\text{target}} - y_{\text{crosshair}})^2}$$

### 5. Reaction Latency
Records initial target spawn timestamp. When crosshair enters the target's bounding box:
```python
if (x1 <= cx_screen <= x2) and (y1 <= cy_screen <= y2):
    rt_ms = (now - self.spawn_timestamps[target_id]) * 1000.0
```

### 6. Overshoot Detection
Tracks horizontal delta ($dx = \text{target}_x - \text{crosshair}_x$). Detects rapid sign flips within 60 px of target center:
```python
if (prev_dx < -12 and dx > 12) or (prev_dx > 12 and dx < -12):
    self.overshoot_count += 1
```

---

## F. `capture_screen_region()` – Screen Capture

Uses `mss` to capture a center ROI (e.g. 960 × 540) around the crosshair:
- Drastically reduces pixel volume compared to a full 1440p/4K display.
- Maximizes capture and inference frame rates.

---

# 3. `telemetry_app.py` – The Non-Blocking Dashboard

The dashboard provides a 3-panel Tkinter layout:
- **Top Bar**: Mode selector (Live Screen vs VOD), file chooser, Start/Stop toggle.
- **Left Panel**: 60 FPS annotated video feed.
- **Right Panel**: Telemetry metric cards, API baseline benchmark sync, and diagnostic event feed.

### Concurrency & Thread Safety
- Capture and inference run on a background daemon worker thread (`_capture_worker`).
- Telemetry state is protected by a thread `Lock`.
- Tkinter GUI updates occur safely on the main thread via `root.after(16, self._schedule_ui_refresh)`.
- Benchmark comparisons run asynchronously on worker threads to avoid UI stutter.

---

# 4. Swapping In a Custom Neural Detector

When ready to replace `ColorTargetTracker`:

1. Export your model (YOLO, RT-DETR, RF-DETR) to ONNX format (`model.onnx`).
2. Implement or configure `ONNXCustomTracker(BaseMechanicTracker)` in `vision_engine.py`:
   ```python
   class ONNXCustomTracker(BaseMechanicTracker):
       def __init__(self, model_path: str):
           import onnxruntime as ort
           self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

       def detect_and_track(self, frame: np.ndarray) -> List[TrackedEntity]:
           # 1. Preprocess frame
           # 2. Run inference: outputs = self.session.run(...)
           # 3. NMS & confidence thresholding
           # 4. Return List[TrackedEntity]
   ```
3. Update one line in `telemetry_app.py`:
   ```python
   # self.tracker = ColorTargetTracker()
   self.tracker = ONNXCustomTracker("models/detector.onnx")
   ```

---

# 5. Known Limitations & Improvements

| Area | Current Behavior | Recommended Upgrade |
|---|---|---|
| **Tracking** | Nearest-centroid distance threshold (85px) | Implement ByteTrack or Kalman-filter based SORT |
| **Reaction Latency** | Measures system detection to acquisition | Explicit target spawn/despawn event detection |
| **Head Drift** | Target bounding box center vs horizon | Multi-keypoint / head-specific detection |
| **Overshoot** | 1D horizontal sign inversion | 2D velocity vector reversal detection |
| **Persistence** | In-memory session stats | SQLite local telemetry storage |
| **API Sync** | Rule-based mock diagnostic flags | Real REST/FastAPI integration for player stats |

---

# 6. Development Roadmap

- **Phase 1 (Foundation - Completed)**: Decoupled `vision_engine.py` + `telemetry_app.py`, HSV tracker, mss screen capture, VOD loader.
- **Phase 2 (Robust Tracking)**: Multi-object lifecycle management, despawn cleanup, ByteTrack integration.
- **Phase 3 (Neural Models)**: ONNX Runtime GPU inference (RT-DETR / YOLO), automated model benchmarking.
- **Phase 4 (Session Analytics & API)**: SQLite persistent session logs, heatmaps, habit trend charts, and REST API baseline synchronization.
