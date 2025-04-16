# import cv2
# import numpy as np

# # Use device camera
# cap = cv2.VideoCapture(0)

# # Read first frame
# ret, prev_frame = cap.read()
# prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# # Initialize heatmap accumulator
# heatmap_accumulator = np.zeros_like(prev_frame, dtype=np.float32)

# # Decay factor
# decay_factor = 0.99

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Compute absolute difference
#     frame_diff = cv2.absdiff(prev_frame, gray_frame)

#     # Accumulate differences
#     heatmap_accumulator += frame_diff

#     # Apply decay to the heatmap accumulator
#     heatmap_accumulator *= decay_factor

#     # Normalize heatmap
#     heatmap_normalized = cv2.normalize(
#         heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX)

#     # Apply colormap
#     heatmap_colored = cv2.applyColorMap(
#         heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

#     # Display heatmap
#     cv2.imshow("Heatmap", heatmap_colored)

#     # Find contours using the heatmap data
#     heatmap_uint8 = heatmap_normalized.astype(np.uint8)
#     contours, _ = cv2.findContours(
#         heatmap_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Create a copy of the original frame to draw rectangles
#     frame_with_boxes = frame.copy()

#     for contour in contours:
#         if cv2.contourArea(contour) > 500:  # Filter out small contours
#             x, y, w, h = cv2.boundingRect(contour)
#             # Check if the area is not blue (low heat)
#             # Adjust threshold as needed
#             if np.mean(heatmap_colored[y:y+h, x:x+w]) > 50:
#                 cv2.rectangle(frame_with_boxes, (x, y),
#                               (x + w, y + h), (0, 255, 0), 2)

#     # Display frame with rectangles
#     cv2.imshow("Frame with Rectangles", frame_with_boxes)

#     # Break loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     prev_frame = gray_frame

# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import numpy as np

# # Replace 'your_video.mp4' with the actual path to your video file
# video_path = 'Video by Abed Ismail from Pexels: https://www.pexels.com/video/a-car-drifting-on-a-racing-track-4568686/'  # Or use a raw string: r'C:\path\to\your_video.mp4'
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print(f"Error: Could not open video file: {video_path}")
#     exit()

# background_subtractor = cv2.createBackgroundSubtractorMOG2()  # Or KNNBackgroundSubtractor

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # End of video

#     # Apply background subtraction
#     fg_mask = background_subtractor.apply(frame)

#     # Noise reduction (Gaussian blur)
#     fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

#     # Thresholding (if needed after background subtraction)
#     _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)


#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     frame_with_boxes = frame.copy()

#     for contour in contours:
#         if cv2.contourArea(contour) > 500:  # Adjust this threshold
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     cv2.imshow("Foreground Mask", fg_mask)  # Show the result of background subtraction
#     cv2.imshow("Frame with Rectangles", frame_with_boxes)

#     if cv2.waitKey(25) & 0xFF == ord('q'):  # Adjust delay (25ms) for playback speed
#         break

# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import numpy as np

# # video_path = 'your_video.mp4'
# cap = cv2.VideoCapture(0)

# # if not cap.isOpened():
# #     print(f"Error: Could not open video file: {video_path}")
# #     exit()

# background_subtractor = cv2.createBackgroundSubtractorMOG2()  # Or KNNBackgroundSubtractor

# # Heatmap parameters
# heatmap_accumulator = None  # Initialize later based on frame size
# decay_factor = 0.95        # Adjust this (closer to 1 for slower decay)
# accumulation_factor = 0.1 # How much new motion contributes (adjust this)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     fg_mask = background_subtractor.apply(frame)
#     fg_mask = cv2.GaussianBlur(fg_mask, (3, 3), 0) # Smaller blur for faster motion
#     _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

#     # Initialize heatmap on the first frame
#     if heatmap_accumulator is None:
#         heatmap_accumulator = np.zeros_like(thresh, dtype=np.float32)

#     # Accumulate motion into the heatmap (weighted)
#     heatmap_accumulator = heatmap_accumulator * decay_factor + thresh * accumulation_factor

#     # Normalize heatmap for display
#     heatmap_normalized = cv2.normalize(heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     frame_with_boxes = frame.copy()

#     for contour in contours:
#         if cv2.contourArea(contour) > 200: # Adjust this
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     cv2.imshow("Heatmap", heatmap_colored) # Show the heatmap
#     cv2.imshow("Foreground Mask", thresh) # Show the foreground mask
#     cv2.imshow("Frame with Rectangles", frame_with_boxes)

#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




# NON-HUMAN ACTIVITIES


# import cv2
# import numpy as np

# # video_path = r'c:\Users\Dibyadyuti Chowdhury\Downloads\4568686-hd_1920_1080_30fps (1).mp4'
# cap = cv2.VideoCapture(0)

# # if not cap.isOpened():
# #     print(f"Error: Could not open video file: {video_path}")
# #     exit()

# background_subtractor = cv2.createBackgroundSubtractorMOG2()

# # Heatmap parameters (adjust these)
# heatmap_accumulator = None
# decay_factor = 0.8
# accumulation_factor = 1

# # Threshold for "large-scale" movement (adjust this)
# large_movement_threshold = 5000  # Example value; experiment!

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     fg_mask = background_subtractor.apply(frame)
#     fg_mask = cv2.GaussianBlur(fg_mask, (3, 3), 0)
#     _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)


#     if heatmap_accumulator is None:
#         heatmap_accumulator = np.zeros_like(thresh, dtype=np.float32)

#     heatmap_accumulator = heatmap_accumulator * decay_factor + thresh * accumulation_factor
#     heatmap_normalized = cv2.normalize(heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)


#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     frame_with_boxes = frame.copy()  # Always copy the frame

#     large_movement_detected = False  # Flag for large movement

#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > 200: # Original threshold
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         if area > large_movement_threshold:  # Check for large movement
#             large_movement_detected = True
#             # You can add additional actions here for large movements
#             # For Example, you could highlight the bounding boxes in a different color
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 3) # Red boxes

#     # Conditional display (only show frame_with_boxes if large movement detected)
#     if large_movement_detected:
#         cv2.imshow("Heatmap", heatmap_colored)
#         cv2.imshow("Foreground Mask", thresh)
#         cv2.imshow("Frame with Rectangles", frame_with_boxes)
#     else:
#         cv2.imshow("Heatmap", heatmap_colored)
#         cv2.imshow("Foreground Mask", thresh)
#         cv2.imshow("Frame with Rectangles", frame) # Show original frame if no large motion


#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





# ______________________________________________________________________________________ #
# CODE TO BE MODIFIED


# import numpy as np
# import cv2

# # Load class labels
# with open(r'./python/action_recognition_kinetics.txt') as f:
#     class_labels = f.read().strip().split('\n')

# # Load the pre-trained model
# net = cv2.dnn.readNet(r'./python/resnet-34_kinetics.onnx')

# # Specify the path to your video file
# #first_video_path = r'C:\Users\Dibyadyuti Chowdhury\Downloads\8986885-uhd_3840_2160_30fps.mp4'
# video_path = r"C:\Users\Dibyadyuti Chowdhury\Downloads\5151347-uhd_2560_1440_30fps.mp4"

# cap = cv2.VideoCapture(video_path)

# # Check if the video file opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Frame dimensions and parameters
# sample_duration = 16  # Number of frames to sample
# sample_size = 112     # Size to which frames are resized

# frames = []

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or error reading frame.")
#         break

#     # Resize frame
#     resized_frame = cv2.resize(frame, (sample_size, sample_size))
#     frames.append(resized_frame)

#     # Ensure we have the required number of frames
#     if len(frames) < sample_duration:
#         continue

#     # Prepare the blob from the frames
#     blob = cv2.dnn.blobFromImages(frames, 1.0,
#                                   (sample_size, sample_size),
#                                   (114.7748, 107.7354, 99.4750),
#                                   swapRB=True, crop=True)
#     blob = np.transpose(blob, (1, 0, 2, 3))
#     blob = np.expand_dims(blob, axis=0)

#     # Set the input and perform a forward pass
#     net.setInput(blob)
#     outputs = net.forward()
#     prediction = outputs[0].argmax()
#     label = class_labels[prediction]

#     # Display the label on the frame
#     cv2.putText(frame, f'Activity: {label}', (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Show the frame
#     cv2.imshow('Human Activity Recognition', frame)

#     # Remove the first frame from the buffer
#     frames.pop(0)

#     # Break on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()




# import numpy as np
# import cv2

# # Load class labels
# with open(r'./python/action_recognition_kinetics.txt') as f:
#     class_labels = f.read().strip().split('\n')

# # Load the pre-trained model
# net = cv2.dnn.readNet(r'./python/resnet-34_kinetics.onnx')

# # Specify the video file path
# video_path = r"C:\Users\Dibyadyuti Chowdhury\Downloads\5151347-uhd_2560_1440_30fps.mp4"
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Frame processing parameters
# sample_duration = 16  # Number of frames to sample
# sample_size = 112     # Size to which frames are resized
# frames = []

# # Define relevant actions that should be marked
# relevant_actions = {"jumping", "waving", "walking", "running", "clapping"}  # Modify as needed

# # Read the first frame to initialize motion detection
# ret, prev_frame = cap.read()
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if ret else None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or error reading frame.")
#         break
    
#     # Resize frame for model input
#     resized_frame = cv2.resize(frame, (sample_size, sample_size))
#     frames.append(resized_frame)
    
#     # Ensure we have enough frames for action recognition
#     if len(frames) < sample_duration:
#         prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         continue
    
#     # Prepare blob for model input
#     blob = cv2.dnn.blobFromImages(frames, 1.0, (sample_size, sample_size),
#                                   (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
#     blob = np.transpose(blob, (1, 0, 2, 3))
#     blob = np.expand_dims(blob, axis=0)
    
#     # Predict activity
#     net.setInput(blob)
#     outputs = net.forward()
#     prediction = outputs[0].argmax()
#     label = class_labels[prediction]
    
#     # Convert to grayscale for motion detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_diff = cv2.absdiff(prev_gray, gray) if prev_gray is not None else np.zeros_like(gray)
#     _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
#     # Check if the detected activity is relevant
#     if label in relevant_actions:
#         # Find contours of moving areas
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in contours:
#             if cv2.contourArea(cnt) > 500:  # Ignore small movements
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
#     # Display activity label
#     cv2.putText(frame, f'Activity: {label}', (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
#     # Show the frame
#     cv2.imshow('Human Activity Recognition', frame)
    
#     # Remove the first frame from the buffer
#     frames.pop(0)
    
#     # Update previous frame for motion detection
#     prev_gray = gray
    
#     # Break on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





# import numpy as np
# import cv2

# # Load class labels
# with open(r'./python/action_recognition_kinetics.txt') as f:
#     class_labels = f.read().strip().split('\n')

# # Load the pre-trained model
# net = cv2.dnn.readNet(r'./python/resnet-34_kinetics.onnx')

# # Specify the video file path
# video_path = r"C:\Users\Dibyadyuti Chowdhury\Downloads\4761738-uhd_4096_2160_25fps.mp4"
# cap = cv2.VideoCapture(video_path)


# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Frame processing parameters
# sample_duration = 16  # Number of frames to sample
# sample_size = 112     # Size to which frames are resized
# frames = []

# # Define relevant actions that should be marked
# relevant_actions = {"jumping", "waving", "walking", "running", "clapping"}  # Modify as needed

# # Read the first frame to initialize motion detection
# ret, prev_frame = cap.read()
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if ret else None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or error reading frame.")
#         break
    
#     # Resize frame for model input
#     resized_frame = cv2.resize(frame, (sample_size, sample_size))
#     frames.append(resized_frame)
    
#     # Ensure we have enough frames for action recognition
#     if len(frames) < sample_duration:
#         prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         continue
    
#     # Prepare blob for model input
#     blob = cv2.dnn.blobFromImages(frames, 1.0, (sample_size, sample_size),
#                                   (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
#     blob = np.transpose(blob, (1, 0, 2, 3))
#     blob = np.expand_dims(blob, axis=0)
    
#     # Predict activity
#     net.setInput(blob)
#     outputs = net.forward()
#     prediction = outputs[0].argmax()
#     label = class_labels[prediction]
    
#     # Convert to grayscale for motion detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_diff = cv2.absdiff(prev_gray, gray) if prev_gray is not None else np.zeros_like(gray)
    
#     # Apply thresholding (higher threshold to ignore minor movements)
#     _, thresh = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
    
#     # Apply morphological operations to remove noise
#     kernel = np.ones((5, 5), np.uint8)
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
#     # Check if the detected activity is relevant
#     if label in relevant_actions:
#         # Find contours of moving areas
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in contours:
#             if cv2.contourArea(cnt) > 2000:  # Increased threshold for major movements
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
#     # Display activity label
#     cv2.putText(frame, f'Activity: {label}', (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
#     # Show the frame
#     cv2.imshow('Human Activity Recognition', frame)
    
#     # Remove the first frame from the buffer
#     frames.pop(0)
    
#     # Update previous frame for motion detection
#     prev_gray = gray
    
#     # Break on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


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
