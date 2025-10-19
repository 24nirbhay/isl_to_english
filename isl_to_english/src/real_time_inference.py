import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from joblib import load

def run_inference():
    # --- 1. Load the new image classification model and class names ---
    model = tf.keras.models.load_model("models/image_gesture_recognizer.keras")
    class_names = load("models/image_class_names.joblib")
    img_size = (224, 224) # Must match the training configuration

    # --- 2. Setup MediaPipe Hands ---
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    # --- 3. Setup OpenCV Video Capture ---
    cap = cv2.VideoCapture(0)

    # --- 4. Real-time Inference Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks for visualization
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- 5. Extract Bounding Box and Crop Hand ---
                h, w, _ = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min: x_min = x
                    if x > x_max: x_max = x
                    if y < y_min: y_min = y
                    if y > y_max: y_max = y

                # Add some padding to the bounding box
                padding = 30
                x_min -= padding
                y_min -= padding
                x_max += padding
                y_max += padding

                # Crop the hand image
                hand_img = frame[y_min:y_max, x_min:x_max]

                # --- 6. Preprocess and Predict ---
                if hand_img.size > 0:
                    img_resized = cv2.resize(hand_img, img_size)
                    img_array = tf.expand_dims(img_resized, 0) # Create a batch
                    predictions = model.predict(img_array)
                    predicted_class = class_names[np.argmax(predictions[0])]
                    confidence = 100 * np.max(predictions[0])

                    # Display the prediction on the frame
                    text = f"{predicted_class} ({confidence:.2f}%)"
                    cv2.putText(frame, text, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("ISL to English Translator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_inference()
