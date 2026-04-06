import cv2
import sys
import os
import mediapipe as mp
from datetime import datetime
import numpy as np

# For the new MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Define hand connections manually
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_landmarks_on_image(rgb_image, detection_result):
    """A helper function to draw landmarks on the image."""
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to draw landmarks
    for hand_landmarks in hand_landmarks_list:
        # Draw landmarks as circles
        for landmark in hand_landmarks:
            x = int(landmark.x * annotated_image.shape[1])
            y = int(landmark.y * annotated_image.shape[0])
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_x = int(hand_landmarks[start_idx].x * annotated_image.shape[1])
            start_y = int(hand_landmarks[start_idx].y * annotated_image.shape[0])
            end_x = int(hand_landmarks[end_idx].x * annotated_image.shape[1])
            end_y = int(hand_landmarks[end_idx].y * annotated_image.shape[0])
            cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
    
    return annotated_image
def main():
    """
    Main function to run the face detection application.
    """

    # Load the pre-trained Haar Cascade for face detection.
    # The XML file is bundled with the OpenCV library.
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        print(f"Error loading cascade file: {e}")
        sys.exit(1)

    # --- Initialize MediaPipe Hand Landmarker using the new Tasks API ---
    model_path = 'hand_landmarker.task'
    if not os.path.exists(model_path):
        print(f"Error: Hand landmarker model not found at {model_path}")
        print("Please download the model from https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        sys.exit(1)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Create an output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize video capture from the default webcam (index 0).
    cap = cv2.VideoCapture(0)

    # Check if the webcam was opened successfully.
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("Starting webcam feed. Press 'q' to exit. Press 's' to save a snapshot.")

    frame_timestamp_ms = 0
    while True:
        # Capture frame-by-frame. ret is a boolean that is False if the frame is not read correctly.
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting...")
            break
        frame_timestamp_ms += 1 # Simple timestamp for video mode

        # Flip the frame horizontally for a later selfie-view display
        # This also makes hand tracking logic easier
        frame = cv2.flip(frame, 1)

        # Convert the frame to grayscale for the face detection algorithm.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame.
        # These parameters can be tuned for performance vs. accuracy.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw a rectangle around each detected face.
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        hand_results = detector.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw hand landmarks if detected
        if hand_results.hand_landmarks:
            annotated_image = draw_landmarks_on_image(rgb_frame, hand_results)
            frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # --- Finger Counting Logic ---
        if hand_results.hand_landmarks:
            for hand_landmarks in hand_results.hand_landmarks:
                # Get landmark coordinates
                landmarks = hand_landmarks
                tip_ids = [4, 8, 12, 16, 20]  # Landmark indices for fingertips
                fingers = []

                # Thumb: Check if tip's x is to the left of the joint below it
                if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other 4 fingers: Check if tip's y is above the joint two landmarks below
                for id in range(1, 5):
                    if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                total_fingers = fingers.count(1)
                # Display the finger count
                cv2.putText(frame, f'Fingers: {total_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame in a window.
        cv2.imshow('Face Detection', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        # Break the loop and exit when 'q' is pressed.
        if key == ord('q'):
            break
        # Save the current frame when 's' is pressed
        elif key == ord('s'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(output_dir, f'snapshot_{timestamp}.png')
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved as {filename}")

    # When everything is done, release the capture and destroy all windows.
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)