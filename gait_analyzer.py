import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime

def get_next_person_name(results_file):
    if not os.path.isfile(results_file):
        return "Person_1"
    
    with open(results_file, "r") as file:
        rows = file.readlines()
        if len(rows) <= 1:
            return "Person_1"
        last_row = rows[-1].split(",")[0]
        if last_row.startswith("Person_"):
            num = int(last_row.split("_")[1])
            return f"Person_{num+1}"
        else:
            return "Person_1"

def process_video(video_path, person_name=None, save_output=False):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Video info
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)

    # Optional output video writer
    out = None
    if save_output:
        out = cv2.VideoWriter("skeleton_output.avi",
                              cv2.VideoWriter_fourcc(*"XVID"),
                              fps, (frame_width, frame_height))

    frame_count = 0
    y_positions = []

    cv2.namedWindow("Original Video with Skeleton", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Skeleton on Black Background", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        skeleton_frame = np.zeros_like(frame)

        if results.pose_landmarks:
            # Draw on original
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            # Draw on black background
            mp_drawing.draw_landmarks(
                skeleton_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            # Track ankle Y position
            right_ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            y_positions.append(right_ankle_y)

        frame_count += 1

        cv2.imshow("Original Video with Skeleton", frame)
        cv2.imshow("Skeleton on Black Background", skeleton_frame)

        if save_output and out:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Metrics
    y_positions = np.array(y_positions)
    if len(y_positions) > 1:
        stride_cycles = np.sum((y_positions[1:] - y_positions[:-1]) > 0.02)
        stride_frequency = stride_cycles / (frame_count / fps)
        step_variability = np.std(np.diff(y_positions))
    else:
        stride_cycles = 0
        stride_frequency = 0
        step_variability = 0

    # Save results
    results_file = "gait_results.csv"
    file_exists = os.path.isfile(results_file)
    if not person_name:
        person_name = get_next_person_name(results_file)

    with open(results_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Person", "Date", "Time", "Frames", "Stride Cycles", "Stride Frequency", "Step Variability"])
        now = datetime.now()
        writer.writerow([person_name, now.date(), now.strftime("%H:%M:%S"),
                         frame_count, stride_cycles, round(stride_frequency, 2), round(step_variability, 3)])

    print("\n=== SKELETON GAIT ANALYSIS SUMMARY ===")
    print(f"Person: {person_name}")
    print(f"Frames processed: {frame_count}")
    print(f"Stride cycles detected: {stride_cycles}")
    print(f"Stride frequency (cycles/sec): {stride_frequency:.2f}")
    print(f"Step variability: {step_variability:.3f}")

# Example usage:
process_video("your_video.mp4", save_output=True)
