import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import cv2
import mss
import numpy as np


@dataclass
class TrackedEntity:
    track_id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]


@dataclass
class FrameTelemetry:
    fps: float = 0.0
    active_targets: int = 0
    crosshair_drift_px: float = 0.0       # Vertical drift from calibrated head-level line
    aim_offset_px: float = 0.0            # Distance from screen center (crosshair) to nearest target
    avg_reaction_ms: float = 0.0          # Time from target spawn to crosshair lock
    overshoot_count: int = 0              # Times crosshair crossed past target center rapidly
    annotated_frame: Optional[np.ndarray] = None
    event_log: Optional[str] = None


class BaseMechanicTracker(ABC):
    """Subclass this to plug in any custom ONNX, TensorRT, or PyTorch model."""

    @abstractmethod
    def detect_and_track(self, frame: np.ndarray) -> List[TrackedEntity]:
        pass


class ColorTargetTracker(BaseMechanicTracker):
    """
    Lightweight HSV + Euclidean ID tracker for immediate testing on aim trainers,
    enemy outlines (e.g., purple/red/yellow highlights), or HUD elements.
    """

    def __init__(self, lower_hsv=(135, 100, 100), upper_hsv=(165, 255, 255), min_area=180):
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.min_area = min_area
        self._next_id = 1
        self._prev_centroids: Dict[int, Tuple[int, int]] = {}

    def detect_and_track(self, frame: np.ndarray) -> List[TrackedEntity]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            detections.append((x, y, x + w, y + h, cx, cy))

        # Simple nearest-centroid ID persistence
        tracked: List[TrackedEntity] = []
        updated_centroids: Dict[int, Tuple[int, int]] = {}

        for (x1, y1, x2, y2, cx, cy) in detections:
            matched_id = None
            best_dist = 85.0
            for tid, (px, py) in self._prev_centroids.items():
                dist = math.hypot(cx - px, cy - py)
                if dist < best_dist:
                    best_dist = dist
                    matched_id = tid

            if matched_id is None:
                matched_id = self._next_id
                self._next_id += 1

            updated_centroids[matched_id] = (cx, cy)
            tracked.append(
                TrackedEntity(
                    track_id=matched_id,
                    label="target",
                    confidence=0.95,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                )
            )

        self._prev_centroids = updated_centroids
        return tracked


class ONNXCustomTracker(BaseMechanicTracker):
    """
    Drop-in ONNX detector interface for custom-trained models (e.g., RT-DETR, YOLO, RF-DETR).
    Requires onnxruntime installed (pip install onnxruntime or onnxruntime-gpu).
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = None

        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
        except ImportError:
            self.session = None
        except Exception as e:
            print(f"[ONNXCustomTracker] Warning: Failed to load ONNX model ({model_path}): {e}")

    def detect_and_track(self, frame: np.ndarray) -> List[TrackedEntity]:
        if self.session is None:
            return []

        # Example preprocessing: resize & normalize
        # (Customize according to your specific model's input specs)
        h, w = frame.shape[:2]
        # Placeholder for inference output parsing:
        # 1. Preprocess: cv2.resize, normalize, transpose (1, C, H, W)
        # 2. outputs = self.session.run(None, {self.input_name: blob})
        # 3. Postprocess: filter by confidence, apply cv2.dnn.NMSBoxes
        # 4. Return List[TrackedEntity]
        return []


class MechanicsAnalyzer:
    """Computes reaction latency, crosshair placement drift, and overshoot from tracked entities."""

    def __init__(self, tracker: BaseMechanicTracker, head_level_ratio: float = 0.48):
        self.tracker = tracker
        self.head_level_ratio = head_level_ratio  # Normalized Y coordinate of head-height horizon

        self.spawn_timestamps: Dict[int, float] = {}
        self.locked_ids: set = set()
        self.reaction_times_ms: List[float] = []
        self.overshoot_count = 0
        self._prev_signed_dx: Dict[int, float] = {}
        self._prev_time = time.perf_counter()

    def process_frame(self, frame: np.ndarray) -> FrameTelemetry:
        now = time.perf_counter()
        dt = max(now - self._prev_time, 1e-6)
        self._prev_time = now
        fps = 1.0 / dt

        h, w = frame.shape[:2]
        cx_screen, cy_screen = w // 2, h // 2
        head_y = int(h * self.head_level_ratio)

        entities = self.tracker.detect_and_track(frame)
        annotated = frame.copy()
        event_msg = None

        # Draw crosshair center & calibrated head-level horizon
        cv2.line(annotated, (0, head_y), (w, head_y), (255, 180, 0), 1)
        cv2.drawMarker(
            annotated, (cx_screen, cy_screen), (0, 255, 0),
            markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2
        )

        nearest_offset = 0.0
        vertical_drift = 0.0

        if entities:
            # Focus telemetry on the target closest to the crosshair
            primary = min(
                entities,
                key=lambda e: math.hypot(e.center[0] - cx_screen, e.center[1] - cy_screen)
            )

            if primary.track_id not in self.spawn_timestamps:
                self.spawn_timestamps[primary.track_id] = now

            tx, ty = primary.center
            dx = tx - cx_screen
            dy = ty - cy_screen
            nearest_offset = math.hypot(dx, dy)
            vertical_drift = abs(ty - head_y)

            # Check if crosshair entered target bounding box -> record reaction latency
            x1, y1, x2, y2 = primary.bbox
            if (x1 <= cx_screen <= x2) and (y1 <= cy_screen <= y2):
                if primary.track_id not in self.locked_ids:
                    self.locked_ids.add(primary.track_id)
                    rt_ms = (now - self.spawn_timestamps[primary.track_id]) * 1000.0
                    if 80.0 <= rt_ms <= 2500.0:  # Filter out noise
                        self.reaction_times_ms.append(rt_ms)
                        event_msg = f"Target #{primary.track_id} acquired in {rt_ms:.0f} ms"

            # Detect horizontal overshoot (sign flip across target center within close range)
            prev_dx = self._prev_signed_dx.get(primary.track_id)
            if prev_dx is not None and abs(dx) < 60:
                if (prev_dx < -12 and dx > 12) or (prev_dx > 12 and dx < -12):
                    self.overshoot_count += 1
                    event_msg = f"Overshoot detected on Target #{primary.track_id} ({abs(dx):.0f}px past)"
            self._prev_signed_dx[primary.track_id] = dx

            # Render bounding boxes and aim vector
            for ent in entities:
                ex1, ey1, ex2, ey2 = ent.bbox
                color = (0, 255, 120) if ent.track_id == primary.track_id else (180, 180, 180)
                cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), color, 2)
                cv2.putText(
                    annotated, f"ID:{ent.track_id} {ent.label}",
                    (ex1, max(ey1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

            cv2.line(annotated, (cx_screen, cy_screen), primary.center, (0, 136, 255), 2)

        avg_rt = sum(self.reaction_times_ms) / len(self.reaction_times_ms) if self.reaction_times_ms else 0.0

        return FrameTelemetry(
            fps=fps,
            active_targets=len(entities),
            crosshair_drift_px=vertical_drift,
            aim_offset_px=nearest_offset,
            avg_reaction_ms=avg_rt,
            overshoot_count=self.overshoot_count,
            annotated_frame=annotated,
            event_log=event_msg,
        )


def capture_screen_region(sct: mss.mss, monitor_idx: int = 1, crop_center: Optional[Tuple[int, int]] = (960, 540)) -> np.ndarray:
    """
    Captures the primary monitor and optionally crops a center ROI around the crosshair
    to maximize FPS and reduce inference overhead.
    """
    mon = sct.monitors[monitor_idx]
    if crop_center:
        cw, ch = crop_center
        left = mon["left"] + max((mon["width"] - cw) // 2, 0)
        top = mon["top"] + max((mon["height"] - ch) // 2, 0)
        region = {"left": left, "top": top, "width": min(cw, mon["width"]), "height": min(ch, mon["height"])}
    else:
        region = mon

    raw = np.array(sct.grab(region))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
