import cv2
import mediapipe as mp
import time

# Initialize video capture
cap = cv2.VideoCapture('Dog_videos/IMG_5260.MOV')

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

def process_video():
    healthy_start_time = None
    unhealthy_start_time = None
    last_gait_status = None

    # Variables to accumulate total time in each state
    total_healthy_time = 0
    total_unhealthy_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing (optional)
        frame_resized = cv2.resize(frame, (640, 480))

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Process the image to identify pose landmarks
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            # Check if the spine is horizontal
            if is_spine_horizontal(result.pose_landmarks.landmark):
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(frame_resized, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame_resized, "Spine: Horizontal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Perform gait analysis and display the result
                gait_status = analyze_gait(result.pose_landmarks.landmark)
                cv2.putText(frame_resized, f'Gait: {gait_status}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Track the time spent in each gait status
                if gait_status == "Healthy":
                    if last_gait_status != "Healthy":
                        # If switching to healthy gait, record time spent in previous state
                        if unhealthy_start_time:
                            unhealthy_duration = time.time() - unhealthy_start_time
                            total_unhealthy_time += unhealthy_duration
                        healthy_start_time = time.time()
                    last_gait_status = "Healthy"
                
                elif gait_status == "Injured":
                    if last_gait_status != "Injured":
                        # If switching to injured gait, record time spent in previous state
                        if healthy_start_time:
                            healthy_duration = time.time() - healthy_start_time
                            total_healthy_time += healthy_duration
                        unhealthy_start_time = time.time()
                    last_gait_status = "Injured"

            else:
                # Indicate that the spine is not horizontal
                cv2.putText(frame_resized, "Spine: Not Horizontal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the video frame
        cv2.imshow('Dog Spine and Gait Analysis', frame_resized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Final update when video ends
    if healthy_start_time:
        healthy_duration = time.time() - healthy_start_time
        total_healthy_time += healthy_duration

    if unhealthy_start_time:
        unhealthy_duration = time.time() - unhealthy_start_time
        total_unhealthy_time += unhealthy_duration

    # Print total time spent in each gait state
    print(f"\nTotal Healthy Gait Duration: {total_healthy_time:.2f} seconds")
    print(f"Total Unhealthy Gait Duration: {total_unhealthy_time:.2f} seconds")

    # final determination
    total_time = total_healthy_time + total_unhealthy_time
    try:
        unhealthy_fraction = total_unhealthy_time / total_time
    except ZeroDivisionError:
        print("No dog was detected. Please try again.")
        exit()
    else:
        if unhealthy_fraction <= 0.2:
            print('The dog is healthy')
        else:
            print('The dog is injured. Please see a veterinarian for further review.')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
