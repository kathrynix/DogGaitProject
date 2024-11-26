import cv2
import mediapipe as mp
import time
import tkinter as tk
from tkinter import messagebox

# Initialize video capture
cap = cv2.VideoCapture('Dog_videos/Mala.MOV')
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils

def is_spine_horizontal(landmarks):
    # Using LEFT_SHOULDER, LEFT_HIP, RIGHT_SHOULDER, and RIGHT_HIP for a better estimate of spine alignment
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Calculate horizontal and vertical distances for both sides
    vertical_spine_dist_left = abs(left_shoulder.y - left_hip.y)
    horizontal_spine_dist_left = abs(left_shoulder.x - left_hip.x)

    vertical_spine_dist_right = abs(right_shoulder.y - right_hip.y)
    horizontal_spine_dist_right = abs(right_shoulder.x - right_hip.x)

    # Return True if spine is approximately horizontal for both sides
    return (horizontal_spine_dist_left > vertical_spine_dist_left) and (horizontal_spine_dist_right > vertical_spine_dist_right)

def analyze_gait(landmarks):
    # Simple gait analysis: check for uneven leg height
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

    # Arbitrary threshold for demo purposes (you can adjust this threshold)
    leg_diff_threshold = 0.1  # You might want to tune this threshold

    # Check if legs are uneven and label the gait
    return "Injured" if abs(left_knee.y - right_knee.y) > leg_diff_threshold else "Healthy"

def show_custom_popup(total_healthy_time, total_unhealthy_time, total_spine_horizontal_time):
    # Create the Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window, only show the messagebox

    # Prepare the message for the popup
    result_message = f"Total Healthy Gait Duration: {total_healthy_time:.2f} seconds\n"
    result_message += f"Total Unhealthy Gait Duration: {total_unhealthy_time:.2f} seconds\n"
    result_message += f"Total Time Dog Detected: {total_spine_horizontal_time:.2f} seconds\n\n"

    if total_spine_horizontal_time < 5:
        result_message += "Inconclusive: Dog must be detected for a longer period."
    else:
        total_time = total_healthy_time + total_unhealthy_time
        if total_time > 0:
            unhealthy_fraction = total_unhealthy_time / total_time
            if unhealthy_fraction <= 0.2:
                result_message += 'The dog is healthy'
            else:
                result_message += 'The dog is injured. See a veterinarian.'
        else:
            result_message += "No dog detected long enough for analysis."

    # Show the messagebox with the analysis results
    messagebox.showinfo("Health Analysis Results", result_message)

def process_video():
    healthy_start_time = None
    unhealthy_start_time = None
    last_gait_status = None

    total_healthy_time = 0
    total_unhealthy_time = 0
    total_spine_horizontal_time = 0

    spine_horizontal_start_time = None
    frame_count = 0

    while cap.isOpened():
        frame_count += 1
        current_time = frame_count / fps  # Time in seconds based on frames
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            if is_spine_horizontal(result.pose_landmarks.landmark):
                if spine_horizontal_start_time is None:
                    spine_horizontal_start_time = current_time
                mp_drawing.draw_landmarks(frame_resized, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame_resized, "Spine: Horizontal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                gait_status = analyze_gait(result.pose_landmarks.landmark)
                cv2.putText(frame_resized, f'Gait: {gait_status}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if gait_status == "Healthy":
                    if healthy_start_time is None:
                        healthy_start_time = current_time
                    if unhealthy_start_time is not None:
                        total_unhealthy_time += current_time - unhealthy_start_time
                        unhealthy_start_time = None
                elif gait_status == "Injured":
                    if unhealthy_start_time is None:
                        unhealthy_start_time = current_time
                    if healthy_start_time is not None:
                        total_healthy_time += current_time - healthy_start_time
                        healthy_start_time = None

                last_gait_status = gait_status
            else:
                if spine_horizontal_start_time is not None:
                    total_spine_horizontal_time += current_time - spine_horizontal_start_time
                    spine_horizontal_start_time = None

                if healthy_start_time is not None:
                    total_healthy_time += current_time - healthy_start_time
                    healthy_start_time = None

                if unhealthy_start_time is not None:
                    total_unhealthy_time += current_time - unhealthy_start_time
                    unhealthy_start_time = None
                cv2.putText(frame_resized, "Spine: Not Horizontal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            # If no landmarks detected, reset spine and gait timings
            if spine_horizontal_start_time is not None:
                total_spine_horizontal_time += current_time - spine_horizontal_start_time
                spine_horizontal_start_time = None

            if healthy_start_time is not None:
                total_healthy_time += current_time - healthy_start_time
                healthy_start_time = None

            if unhealthy_start_time is not None:
                total_unhealthy_time += current_time - unhealthy_start_time
                unhealthy_start_time = None

        cv2.imshow('Dog Spine and Gait Analysis', frame_resized)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Final Time Update
    end_time = current_time
    if healthy_start_time:
        total_healthy_time += end_time - healthy_start_time
    if unhealthy_start_time:
        total_unhealthy_time += end_time - unhealthy_start_time
    if spine_horizontal_start_time:
        total_spine_horizontal_time += end_time - spine_horizontal_start_time

    # Show custom popup with results
    show_custom_popup(total_healthy_time, total_unhealthy_time, total_spine_horizontal_time)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video()
