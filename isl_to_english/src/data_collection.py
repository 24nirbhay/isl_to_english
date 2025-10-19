import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = "data/dataset"

def collect_data(gesture_name, num_samples=10, sequence_length=30):
    cap = cv2.VideoCapture(0)  # Use default backend
    hands = mp_hands.Hands(max_num_hands=1)
    os.makedirs(os.path.join(DATA_PATH, gesture_name), exist_ok=True)
    
    for i in range(num_samples):
        sequence = []
        while len(sequence) < sequence_length:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    sequence.append(landmarks)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Collecting Data", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        file_path = os.path.join(DATA_PATH, gesture_name, f"sequence_{i}.csv")
        with open(file_path, mode='w', newline='') as f:
            csv.writer(f).writerows(sequence)
