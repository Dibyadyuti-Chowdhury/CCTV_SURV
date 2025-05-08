import cv2
import numpy as np
from ultralytics import YOLO
import datetime

# Load the YOLOv8-pose model
model = YOLO('yolov8n-pose.pt')  # Use yolov8s/m/l-pose.pt for better accuracy

# Start video capture (0 for webcam, or provide a file path)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video or webcam.")
    exit()

# Variables to store previous keypoints
prev_keypoints = {}

def calculate_movement(prev, curr):
    """Calculate average Euclidean distance between keypoints."""
    if prev is None or curr is None:
        return 0.0
    distances = []
    for p, c in zip(prev, curr):
        if p[2] > 0.3 and c[2] > 0.3:  # confidence threshold
            dist = np.linalg.norm(np.array(p[:2]) - np.array(c[:2]))
            distances.append(dist)
    return np.mean(distances) if distances else 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)[0]
    annotated_frame = results.plot()

    # Process detections
    alert_triggered = False
    for i, kp in enumerate(results.keypoints.xy):
        keypoints = results.keypoints[i].xy[0].cpu().numpy()
        conf = results.keypoints[i].conf[0].cpu().numpy()

        # Combine x, y, confidence
        keypoints_with_conf = [(x, y, c) for (x, y), c in zip(keypoints, conf)]

        # Calculate movement
        movement = calculate_movement(prev_keypoints.get(i), keypoints_with_conf)

        if movement > 15:  # Movement threshold (tweak for sensitivity)
            alert_triggered = True
            prev_keypoints[i] = keypoints_with_conf  # Update stored keypoints

    # Show alert if movement is significant
    if alert_triggered:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(annotated_frame, f"ALERT: Human Movement Detected! [{timestamp}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the result
    cv2.imshow("Human Movement + Keypoint Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
