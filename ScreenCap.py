# ============================================================
# Face Detection Script
# Copyright (c) 2025 - AsaTyr / AIMusics-Central
# This script processes videos to extract faces (realistic or anime)
# and saves them along with full-frame screenshots.
#
# Requirements:
# - YOLOv8 model for anime detection
# - face_recognition library for realistic faces
# - Ensure that GPU acceleration is enabled for optimal performance
#
# Instructions:
# 1. Set the `top_working_dir` to your working directory.
# 2. Place input videos in the 'input' folder under the working directory.
# 3. Ensure that the YOLO model is placed in the 'model' subfolder.
# ============================================================

import cv2
import face_recognition
from PIL import Image
import os
import shutil
from tqdm import tqdm
from ultralytics import YOLO  # YOLOv8
import importlib.util

# Dependency check
def check_dependencies():
    dependencies = [
        ("cv2", "opencv-python"),
        ("face_recognition", "face_recognition"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
        ("ultralytics", "ultralytics"),
    ]
    missing = []

    for module, package in dependencies:
        if not importlib.util.find_spec(module):
            missing.append(package)

    if missing:
        print("The following dependencies are missing:")
        for package in missing:
            print(f"- {package}")
        print("\nInstall them using the following command:")
        print(f"pip install {' '.join(missing)}")
        exit(1)

check_dependencies()

# Base directories for the working environment
top_working_dir = "/ScreenCap"
input_dir = os.path.join(top_working_dir, "input")
work_dir = os.path.join(top_working_dir, "Work")
output_dir = os.path.join(top_working_dir, "output")

screencaps_dir = os.path.join(work_dir, "screencaps")
faces_dir = os.path.join(work_dir, "faces")
model_dir = os.path.join(top_working_dir, "model")

# Ensure necessary directories exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(work_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(screencaps_dir, exist_ok=True)
os.makedirs(faces_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Load YOLOv8 model for anime detection
yolo_model_path = os.path.join(model_dir, "AniRef40000-l-epoch50.pt")  # Specify the path to the YOLO model
yolo_model = YOLO(yolo_model_path)

def detect_faces_realistic(frame, frame_count):
    """
    Detects realistic faces using the face_recognition library (HOG or CNN).
    The faces are cropped and saved in the 'faces' directory.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    faces = []

    for i, (top, right, bottom, left) in enumerate(face_locations):
        face_image = frame[top:bottom, left:right]
        output_path = os.path.join(faces_dir, f"realistic_face_{frame_count}_{i}.jpg")
        cv2.imwrite(output_path, face_image)
        faces.append(output_path)

    return faces

def detect_faces_anime(frame, frame_count):
    """
    Detects anime characters using YOLOv8.
    The results include bounding boxes for characters, which are cropped
    and saved in the 'faces' directory.
    """
    results = yolo_model(frame)
    detected_faces = []

    for i, box in enumerate(results[0].boxes):  # Extract bounding boxes
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0]
        if conf >= 0.5:  # Confidence threshold for valid detections
            face_image = frame[y1:y2, x1:x2]
            output_path = os.path.join(faces_dir, f"anime_face_{frame_count}_{i}.jpg")
            cv2.imwrite(output_path, face_image)
            detected_faces.append(output_path)

    return detected_faces

def process_videos(mode):
    """
    Main function to process videos and extract faces based on the selected mode.
    Modes:
    - 'Realistic': Uses face_recognition for realistic faces.
    - 'Anime': Uses YOLOv8 for anime character detection.
    
    Full-frame screenshots are saved in the 'screencaps' directory.
    Results are organized by video in the 'output' folder.
    """
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".avi", ".mkv"))]

    if not video_files:
        print("No videos found in the input directory.")
        return

    for video_file in tqdm(video_files, desc="Overall Progress", unit="Video"):
        video_path = os.path.join(input_dir, video_file)

        print(f"\nProcessing video: {video_file}")

        # Clear working directories
        for dir_path in [screencaps_dir, faces_dir]:
            for file in os.listdir(dir_path):
                os.unlink(os.path.join(dir_path, file))

        # Open the video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc=f"Processing: {video_file}", unit="Frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Save full-frame screenshot
                screencap_path = os.path.join(screencaps_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(screencap_path, frame)

                # Detect faces based on the selected mode
                if mode == "Realistic":
                    detect_faces_realistic(frame, frame_count)
                elif mode == "Anime":
                    detect_faces_anime(frame, frame_count)

                frame_count += 1
                pbar.update(1)

        cap.release()

        # Move results to the output directory
        video_output_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
        os.makedirs(video_output_dir, exist_ok=True)

        for dir_name in ["screencaps", "faces"]:
            src_dir = os.path.join(work_dir, dir_name)
            dest_dir = os.path.join(video_output_dir, dir_name)
            shutil.copytree(src_dir, dest_dir)

        print(f"Processing completed: Results saved in {video_output_dir}")

if __name__ == "__main__":
    print("Select mode:")
    print("1: Realistic (HOG/CNN)")
    print("2: Anime (YOLOv8)")
    mode_choice = input("Enter mode (1 or 2): ").strip()

    if mode_choice == "1":
        mode = "Realistic"
    elif mode_choice == "2":
        mode = "Anime"
    else:
        print("Invalid input. Exiting.")
        exit(1)

    print(f"Starting processing in {mode} mode...")
    process_videos(mode)
    print("All videos have been processed.")
