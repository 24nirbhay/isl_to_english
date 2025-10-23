import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from joblib import load
from collections import deque
import time
import os

class SignLanguageTranslator:
    def __init__(self, model_dir="models", gesture_threshold=0.6, pause_threshold=2.0):
        """Initialize the translator with models and parameters.
        
        Args:
            model_dir: Directory containing the model files
            gesture_threshold: Confidence threshold for gesture recognition
            pause_threshold: Time in seconds to wait before considering a sentence complete
        """
        # Load the latest model if model_dir is a directory
        if os.path.isdir(model_dir):
            model_dirs = [d for d in os.listdir(model_dir) if d.startswith('model_')]
            if model_dirs:
                latest_model = max(model_dirs)
                model_dir = os.path.join(model_dir, latest_model)
        
        self.model = tf.keras.models.load_model(os.path.join(model_dir, "model.keras"))
        self.le = load(os.path.join(model_dir, "label_encoder.joblib"))
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Parameters
        self.sequence_length = 30
        self.gesture_threshold = gesture_threshold
        self.pause_threshold = pause_threshold
        
        # State variables
        self.sequence = deque(maxlen=self.sequence_length)
        self.current_sentence = []
        self.last_gesture_time = time.time()
        self.last_prediction = None
        self.consecutive_frames = 0
        
    def _extract_hand_landmarks(self, results):
        """Extract hand landmarks in the correct format."""
        single_hand_len = 21 * 3
        left = [0.0] * single_hand_len
        right = [0.0] * single_hand_len

        if not results.multi_hand_landmarks:
            return left + right

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                           results.multi_handedness):
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if handedness.classification[0].label == 'Left':
                left = landmarks
            else:
                right = landmarks

        return left + right
    
    def _predict_gesture(self):
        """Predict the current gesture from the sequence."""
        if len(self.sequence) < self.sequence_length:
            return None, 0.0
            
        X = np.array([list(self.sequence)], dtype='float32')
        preds = self.model.predict(X, verbose=0)[0]
        idx = np.argmax(preds)
        confidence = preds[idx]
        
        if confidence >= self.gesture_threshold:
            return self.le.inverse_transform([idx])[0], confidence
        return None, confidence
    
    def process_frame(self, frame):
        """Process a single frame and update the state."""
        # Flip frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Extract landmarks and add to sequence
        landmarks = self._extract_hand_landmarks(results)
        self.sequence.append(landmarks)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Predict gesture
        gesture, confidence = self._predict_gesture()
        
        # Update state
        current_time = time.time()
        if gesture:
            if gesture != self.last_prediction:
                self.consecutive_frames = 1
            else:
                self.consecutive_frames += 1
                
            if self.consecutive_frames >= 5:  # Require 5 consecutive same predictions
                if not self.current_sentence or self.current_sentence[-1] != gesture:
                    self.current_sentence.append(gesture)
                    self.last_gesture_time = current_time
                
            self.last_prediction = gesture
        
        # Check for sentence completion
        sentence_complete = False
        if self.current_sentence and (current_time - self.last_gesture_time) >= self.pause_threshold:
            sentence_complete = True
            
        # Draw UI
        self._draw_ui(frame, gesture, confidence, current_time)
        
        return frame, sentence_complete
    
    def _draw_ui(self, frame, gesture, confidence, current_time):
        """Draw the UI elements on the frame."""
        # Draw current gesture and confidence
        if gesture:
            text = f"{gesture} ({confidence*100:.1f}%)"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw current sentence
        if self.current_sentence:
            sentence = " ".join(self.current_sentence)
            cv2.putText(frame, sentence, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw pause timer
        if self.current_sentence:
            time_since_last = current_time - self.last_gesture_time
            if time_since_last < self.pause_threshold:
                progress = int((time_since_last / self.pause_threshold) * 100)
                bar_width = 200
                filled_width = int((progress / 100) * bar_width)
                cv2.rectangle(frame, (10, 90), (10 + bar_width, 100), (0, 0, 255), 2)
                cv2.rectangle(frame, (10, 90), (10 + filled_width, 100), (0, 0, 255), -1)
    
    def get_current_sentence(self):
        """Get the current sentence and reset if completed."""
        if not self.current_sentence:
            return ""
        
        sentence = " ".join(self.current_sentence)
        self.current_sentence = []
        self.last_prediction = None
        self.consecutive_frames = 0
        return sentence
    
    def __del__(self):
        """Clean up resources."""
        self.hands.close()

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    translator = SignLanguageTranslator()
    
    print("\nISL to English Translator")
    print("------------------------")
    print("Press 'q' to quit")
    print("Make signs and pause for 2 seconds to complete a sentence")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Process frame
        frame, sentence_complete = translator.process_frame(frame)
        
        # Show frame
        cv2.imshow('ISL to English Translator', frame)
        
        # Check for sentence completion
        if sentence_complete:
            sentence = translator.get_current_sentence()
            if sentence:
                print(f"\nTranslated: {sentence}")
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()