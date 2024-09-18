# Nicer layout but only detects humans

import cv2
import mediapipe as mp

# Initialize video capture
cap = cv2.VideoCapture('Dog_videos/IMG_5260.MOV')

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils

def analyze_gait(landmarks):
    # Placeholder function to analyze gait
    # Compare positions of legs, calculate angles between joints, etc.
    left_leg = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_leg = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

    # Simple rule: if the legs' movement is significantly uneven, the dog may be limping
    # You can make this much more sophisticated with detailed joint analysis
    if abs(left_leg.y - right_leg.y) > 0.1:  # arbitrary threshold for demo purposes
        return "Injured"
    return "Healthy"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get pose landmarks
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Analyze gait based on landmarks
        status = analyze_gait(result.pose_landmarks.landmark)

        # Display the status on the video
        cv2.putText(frame, f'Status: {status}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the video
    cv2.imshow('Dog Gait Analysis', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
