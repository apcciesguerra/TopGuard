#!/usr/bin/env python3
"""
TopGuard Camera Test
Basic YOLOv8 object detection with USB camera for thesis data collection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import tkinter as tk
from tkinter import ttk, messagebox

# Common resolutions to test
RESOLUTIONS = [
    (640, 480, "640x480"),
    (1280, 720, "1280x720 (HD)"),
    (1920, 1080, "1920x1080 (Full HD)"),
    (2560, 1440, "2560x1440 (2K)"),
    (3840, 2160, "3840x2160 (4K)"),
    (1024, 768, "1024x768"),
    (1280, 960, "1280x960"),
]

# Find available cameras with names
def find_cameras():
    available_cameras = {}
    for i in range(10):  # Check first 10 camera indices
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))

                    # Create descriptive name
                    camera_name = f"Camera {i} - {width}x{height} @ {fps}fps"
                    available_cameras[i] = camera_name
                cap.release()
        except:
            pass
    return available_cameras

# GUI for camera selection and camera controls
class CameraApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TopGuard - Camera Test")
        self.root.geometry("430x360")
        self.camera = None
        self.running = False
        self.model = YOLO("yolov8n.pt")
        self.frame_count = 0
        self.start_time = None

        self.cameras = find_cameras()
        if not self.cameras:
            messagebox.showerror("Error", "No cameras found!")
            self.root.destroy()
            return
        self.camera_indices = list(self.cameras.keys())

        panel = ttk.Frame(self.root, padding=18)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="TopGuard Camera Test", font=("Arial", 16, "bold")).pack(pady=(0, 18))
        ttk.Label(panel, text="Camera:").pack(anchor="w")
        self.camera_dropdown = ttk.Combobox(panel, values=[self.cameras[i] for i in self.camera_indices], state="readonly", width=46)
        self.camera_dropdown.current(0)
        self.camera_dropdown.pack(fill="x", pady=(4, 14))
        ttk.Label(panel, text="Resolution:").pack(anchor="w")
        self.resolution_dropdown = ttk.Combobox(panel, values=[f"{w}x{h} - {label}" for w, h, label in RESOLUTIONS], state="readonly", width=46)
        self.resolution_dropdown.current(2)
        self.resolution_dropdown.pack(fill="x", pady=(4, 18))
        buttons = ttk.Frame(panel)
        buttons.pack(fill="x")
        self.start_button = ttk.Button(buttons, text="Start Camera", command=self.start_camera)
        self.start_button.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.stop_button = ttk.Button(buttons, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_button.pack(side="left", expand=True, fill="x", padx=(5, 0))
        self.status = tk.StringVar(value="Camera stopped")
        ttk.Label(panel, textvariable=self.status, foreground="#555").pack(pady=22)
        ttk.Button(panel, text="Exit", command=self.close).pack(fill="x")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(10, self.update_frame)
        self.root.mainloop()

    def start_camera(self):
        self.stop_camera()
        camera_index = self.camera_indices[self.camera_dropdown.current()]
        width, height, _ = RESOLUTIONS[self.resolution_dropdown.current()]
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        if not self.camera.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {camera_index}.")
            self.camera = None
            return
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()
        self.status.set("Camera running - press Q or use Stop Camera")
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

    def stop_camera(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        cv2.destroyAllWindows()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status.set("Camera stopped")

    def update_frame(self):
        if self.running and self.camera is not None:
            ret, frame = self.camera.read()
            if not ret:
                self.status.set("Camera read failed")
                self.stop_camera()
            else:
                results = self.model(frame, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()
                self.frame_count += 1
                fps = self.frame_count / max(time.time() - self.start_time, 0.001)
                detections = results[0].boxes
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}  |  Frame: {self.frame_count}  |  Detections: {len(detections)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("TopGuard - Camera Test", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop_camera()
        self.root.after(10, self.update_frame)

    def close(self):
        self.stop_camera()
        self.root.destroy()


CameraApp()
