# Detects all movement

import cv2
import numpy as np

def detect_objects(frame, net, layer_names):
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416),
                               swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)
    return outputs

def analyze_dog_gait(outputs, frame_height, frame_width):
    detected_objects = []

    for output in outputs:
        # Iterate over each detection in the current output layer
        for detection in output:
            confidence = detection[4]  # Confidence score is at index 4
            if confidence > 0.5:  # Threshold confidence level
                box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (startX, startY, endX, endY) = box.astype("int")
                detected_objects.append((startX, startY, endX, endY))
    
    return detected_objects

def main(video_path):
    # Load YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # Get output layer names
    layer_names = net.getLayerNames()
    out_layers_indices = net.getUnconnectedOutLayers()

    # Convert out_layers_indices to a list if it's a NumPy array
    if isinstance(out_layers_indices, np.ndarray):
        out_layers_indices = out_layers_indices.flatten().tolist()

    # Handle indexing (convert 1-based to 0-based)
    output_layers = [layer_names[i - 1] for i in out_layers_indices]

    # Open video file
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (frame_height, frame_width) = frame.shape[:2]
        outputs = detect_objects(frame, net, output_layers)
        detected_objects = analyze_dog_gait(outputs, frame_height, frame_width)

        # Display the frame with detected objects
        for (startX, startY, endX, endY) in detected_objects:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "Dog_videos/IMG_5261.MOV"
    main(video_path)