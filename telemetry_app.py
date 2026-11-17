import os
import json
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from typing import Optional

import cv2
import mss
import requests
from PIL import Image, ImageTk

from vision_engine import (
    ColorTargetTracker,
    MechanicsAnalyzer,
    FrameTelemetry,
    capture_screen_region,
)


class TelemetryDashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Aim & Mechanics Telemetry Studio")
        self.root.geometry("1360x780")
        self.root.configure(bg="#0F172A")

        # Pipeline state
        self.running = False
        self.source_mode = tk.StringVar(value="screen")  # "screen" or "vod"
        self.vod_path: Optional[str] = None
        self.latest_telemetry: Optional[FrameTelemetry] = None
        self._lock = threading.Lock()

        self.tracker = ColorTargetTracker()
        self.analyzer = MechanicsAnalyzer(self.tracker, head_level_ratio=0.48)

        self._build_ui()
        self._schedule_ui_refresh()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("Metric.TLabel", background="#1E293B", foreground="#F8FAFC", font=("Consolas", 13, "bold"))
        style.configure("Sub.TLabel", background="#1E293B", foreground="#94A3B8", font=("Segoe UI", 9))

        # Top Control Bar
        top_bar = tk.Frame(self.root, bg="#1E293B", pady=10, padx=16)
        top_bar.pack(fill=tk.X)

        tk.Label(
            top_bar, text="MECHANICS TELEMETRY // LIVE & VOD",
            font=("Segoe UI", 14, "bold"), bg="#1E293B", fg="#38BDF8"
        ).pack(side=tk.LEFT)

        controls = tk.Frame(top_bar, bg="#1E293B")
        controls.pack(side=tk.RIGHT)

        ttk.Radiobutton(controls, text="Live Screen (mss)", variable=self.source_mode, value="screen").pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(controls, text="VOD File", variable=self.source_mode, value="vod").pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Select VOD", command=self._pick_vod).pack(side=tk.LEFT, padx=6)
        self.toggle_btn = ttk.Button(controls, text="Start Capture", command=self._toggle_capture)
        self.toggle_btn.pack(side=tk.LEFT, padx=6)

        # Main Split Area
        body = tk.Frame(self.root, bg="#0F172A", padx=16, pady=12)
        body.pack(fill=tk.BOTH, expand=True)

        # Left: Live Annotated Viewport
        video_card = tk.Frame(body, bg="#1E293B", padx=8, pady=8)
        video_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(video_card, bg="#020617", text="Viewport Idle — Click 'Start Capture'", fg="#64748B", font=("Segoe UI", 12))
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right: Telemetry Cards + API Benchmark Panel
        side_panel = tk.Frame(body, bg="#0F172A", width=420)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(14, 0))
        side_panel.pack_propagate(False)

        self.metric_vars = {
            "fps": tk.StringVar(value="0.0 FPS"),
            "rt": tk.StringVar(value="-- ms"),
            "drift": tk.StringVar(value="-- px"),
            "overshoot": tk.StringVar(value="0"),
        }

        cards_meta = [
            ("CAPTURE RATE", "fps"),
            ("AVG REACTION LATENCY", "rt"),
            ("CROSSHAIR HEAD-DRIFT", "drift"),
            ("OVERSHOOT EVENTS", "overshoot"),
        ]

        for title, key in cards_meta:
            card = tk.Frame(side_panel, bg="#1E293B", padx=12, pady=8)
            card.pack(fill=tk.X, pady=(0, 8))
            ttk.Label(card, text=title, style="Sub.TLabel").pack(anchor=tk.W)
            ttk.Label(card, textvariable=self.metric_vars[key], style="Metric.TLabel").pack(anchor=tk.W, pady=(2, 0))

        # External Stats / Benchmark Sync Box
        api_card = tk.Frame(side_panel, bg="#1E293B", padx=12, pady=10)
        api_card.pack(fill=tk.X, pady=(4, 8))

        tk.Label(api_card, text="PLAYER / BENCHMARK API SYNC", bg="#1E293B", fg="#38BDF8", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.player_id_entry = ttk.Entry(api_card, font=("Segoe UI", 10))
        self.player_id_entry.insert(0, "PlayerTag#0001")
        self.player_id_entry.pack(fill=tk.X, pady=6)

        ttk.Button(api_card, text="Compare Against Baseline API", command=self._fetch_api_baseline).pack(fill=tk.X)

        # Live Mistake & Event Feed
        tk.Label(side_panel, text="LIVE MECHANICS DIAGNOSTICS", bg="#0F172A", fg="#94A3B8", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(4, 2))
        self.log_box = scrolledtext.ScrolledText(
            side_panel, height=14, bg="#020617", fg="#E2E8F0",
            font=("Consolas", 9), relief=tk.FLAT, padx=8, pady=8
        )
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self._append_log("System ready. Select Live Screen or a VOD file.")

    def _pick_vod(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.avi")])
        if path:
            self.vod_path = path
            self.source_mode.set("vod")
            self._append_log(f"Loaded VOD: {os.path.basename(path)}")

    def _toggle_capture(self):
        if self.running:
            self.running = False
            self.toggle_btn.config(text="Start Capture")
            self._append_log("Capture stopped.")
        else:
            if self.source_mode.get() == "vod" and not self.vod_path:
                self._append_log("Error: Pick a VOD file first or switch to Live Screen.")
                return
            self.running = True
            self.toggle_btn.config(text="Stop Capture")
            threading.Thread(target=self._capture_worker, daemon=True).start()
            self._append_log(f"Started capture in [{self.source_mode.get().upper()}] mode.")

    def _capture_worker(self):
        if self.source_mode.get() == "screen":
            with mss.mss() as sct:
                while self.running:
                    frame = capture_screen_region(sct, monitor_idx=1, crop_center=(960, 540))
                    telemetry = self.analyzer.process_frame(frame)
                    with self._lock:
                        self.latest_telemetry = telemetry
        else:
            cap = cv2.VideoCapture(self.vod_path)
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (960, 540))
                telemetry = self.analyzer.process_frame(frame)
                with self._lock:
                    self.latest_telemetry = telemetry
                cv2.waitKey(16)  # ~60 FPS VOD playback cap
            cap.release()
            self.running = False

    def _schedule_ui_refresh(self):
        with self._lock:
            tel = self.latest_telemetry
            self.latest_telemetry = None

        if tel is not None:
            self.metric_vars["fps"].set(f"{tel.fps:.1f} FPS")
            self.metric_vars["rt"].set(f"{tel.avg_reaction_ms:.0f} ms" if tel.avg_reaction_ms > 0 else "-- ms")
            self.metric_vars["drift"].set(f"{tel.crosshair_drift_px:.1f} px")
            self.metric_vars["overshoot"].set(str(tel.overshoot_count))

            if tel.event_log:
                self._append_log(tel.event_log)

            if tel.annotated_frame is not None:
                rgb = cv2.cvtColor(tel.annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb).resize((860, 484), Image.BILINEAR)
                photo = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo

        self.root.after(16, self._schedule_ui_refresh)  # ~60Hz UI poll

    def _fetch_api_baseline(self):
        player_tag = self.player_id_entry.get().strip()
        self._append_log(f"Syncing baseline metrics for {player_tag}...")

        def worker():
            try:
                # Replace URL with your stats endpoint (e.g., HenrikDev Valorant API / custom FastAPI service)
                api_key = os.getenv("GAME_STATS_API_KEY", "")
                headers = {"Authorization": api_key} if api_key else {}
                # Example diagnostic rule comparing live session metrics to target thresholds
                avg_rt = float(self.metric_vars["rt"].get().replace(" ms", "")) if "--" not in self.metric_vars["rt"].get() else 0.0
                overshoots = int(self.metric_vars["overshoot"].get())

                flags = []
                if avg_rt > 260:
                    flags.append(f"Slow acquisition ({avg_rt:.0f}ms vs 220ms target)")
                if overshoots >= 3:
                    flags.append(f"High flick overshoot ({overshoots}x) — lower eDPI or decelerate earlier")
                if not flags:
                    flags.append("Mechanics within benchmark tolerance.")

                summary = " | ".join(flags)
                self.root.after(0, lambda: self._append_log(f"[DIAGNOSTIC] {summary}"))
            except Exception as exc:
                self.root.after(0, lambda: self._append_log(f"[API ERROR] {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _append_log(self, text: str):
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.insert(tk.END, f"> {text}\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)


if __name__ == "__main__":
    app_root = tk.Tk()
    TelemetryDashboard(app_root)
    app_root.mainloop()
