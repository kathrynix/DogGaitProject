import cv2
import mediapipe as mp

# Initialize video capture
cap = cv2.VideoCapture('Dog_videos/IMG_5263.MOV')

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize drawing utilities
# It's responsible for the visual representation of the pose detection (the red dots and white lines)
mp_drawing = mp.solutions.drawing_utils

def is_spine_horizontal(landmarks):
    # Using LEFT_SHOULDER and LEFT_HIP to approximate spine alignment
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

    # Calculate the vertical and horizontal distances
    vertical_spine_dist = abs(left_shoulder.y - left_hip.y)
    horizontal_spine_dist = abs(left_shoulder.x - left_hip.x)

    # Return True if the spine is horizontal
    return horizontal_spine_dist > vertical_spine_dist

def analyze_gait(landmarks):
    # Simple gait analysis: check for uneven leg height
    left_leg = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_leg = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

    # Arbitrary threshold for demo purposes
    return "Injured" if abs(left_leg.y - right_leg.y) > 0.1 else "Healthy"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Image is processed to identify pose landmarks
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        # Check if the spine is horizontal
        if is_spine_horizontal(result.pose_landmarks.landmark):
            # Draw landmarks only if spine is horizontal
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, "Spine: Horizontal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Perform gait analysis and display the result
            gait_status = analyze_gait(result.pose_landmarks.landmark)
            cv2.putText(frame, f'Gait: {gait_status}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # Indicate that the spine is not horizontal
            cv2.putText(frame, "Spine: Not Horizontal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the video
    cv2.imshow('Dog Spine and Gait Analysis', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
