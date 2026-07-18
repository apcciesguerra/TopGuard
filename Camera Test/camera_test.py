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

# GUI for camera and resolution selection
class CameraSelector:
    def __init__(self):
        self.selected_camera = None
        self.selected_resolution = (1920, 1080)  # Default to Full HD
        self.cameras = find_cameras()  # Dict of {index: name}
        self.camera_indices = list(self.cameras.keys())

        if not self.cameras:
            messagebox.showerror("Error", "No cameras found!")
            exit(1)

        self.root = tk.Tk()
        self.root.title("TopGuard - Camera & Resolution Selection")
        self.root.geometry("550x320")

        # Camera Label
        camera_label = ttk.Label(self.root, text="Select Camera:", font=("Arial", 11, "bold"))
        camera_label.pack(pady=(10, 5))

        # Dropdown for camera selection
        self.camera_var = tk.StringVar()
        camera_options = [self.cameras[i] for i in self.camera_indices]
        self.camera_dropdown = ttk.Combobox(
            self.root,
            textvariable=self.camera_var,
            values=camera_options,
            state="readonly",
            width=50
        )
        self.camera_dropdown.pack(pady=5)
        self.camera_dropdown.current(0)  # Default to first camera

        # Resolution Label
        res_label = ttk.Label(self.root, text="Select Resolution:", font=("Arial", 11, "bold"))
        res_label.pack(pady=(15, 5))

        # Dropdown for resolution selection
        self.resolution_var = tk.StringVar()
        res_options = [f"{w}x{h} - {label}" for w, h, label in RESOLUTIONS]
        self.resolution_dropdown = ttk.Combobox(
            self.root,
            textvariable=self.resolution_var,
            values=res_options,
            state="readonly",
            width=50
        )
        self.resolution_dropdown.pack(pady=5)
        self.resolution_dropdown.current(2)  # Default to 1920x1080

        # Start button
        start_button = ttk.Button(self.root, text="Start Camera", command=self.start_camera)
        start_button.pack(pady=15)

        # Exit button
        exit_button = ttk.Button(self.root, text="Exit", command=self.root.quit)
        exit_button.pack(pady=5)

        self.root.mainloop()

    def start_camera(self):
        selected_index = self.camera_dropdown.current()
        self.selected_camera = self.camera_indices[selected_index]

        res_index = self.resolution_dropdown.current()
        self.selected_resolution = (RESOLUTIONS[res_index][0], RESOLUTIONS[res_index][1])

        self.root.destroy()

# Select camera and resolution
selector = CameraSelector()
camera_index = selector.selected_camera
width, height = selector.selected_resolution

# Initialize camera
print(f"Opening camera {camera_index}...")
camera = cv2.VideoCapture(camera_index)

if not camera.isOpened():
    print(f"ERROR: Cannot open camera {camera_index}. Check connection.")
    exit(1)

# Set camera resolution
print(f"Setting resolution to {width}x{height}...")
camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
camera.set(cv2.CAP_PROP_FPS, 30)

# Verify actual resolution
actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual resolution: {actual_width}x{actual_height}")

print("Loading YOLOv8 model...")
# Load YOLOv8 nano model (lightweight, good for testing)
model = YOLO("yolov8n.pt")

print("Camera ready. Press 'q' to quit.")
print("-" * 50)

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("ERROR: Failed to read frame from camera")
            break

        # Run YOLOv8 detection
        results = model(frame, conf=0.5)

        # Draw results on frame
        annotated_frame = results[0].plot()

        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Display FPS on frame
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display frame count
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_count}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Show detections count
        detections = results[0].boxes
        cv2.putText(
            annotated_frame,
            f"Detections: {len(detections)}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display the frame
        cv2.imshow("TopGuard - Camera Test", annotated_frame)

        # Print detection info to console (every 30 frames)
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} | FPS: {fps:.2f} | Detections: {len(detections)}")
            for box in detections:
                class_name = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                print(f"  - {class_name}: {confidence:.2f}")

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nQuitting...")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    print(f"\nTotal frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")
    camera.release()
    cv2.destroyAllWindows()
    print("Camera released. Test complete.")
