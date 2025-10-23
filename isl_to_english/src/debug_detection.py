import cv2
import mediapipe as mp
import os
from data_collection import extract_landmarks_from_image_file

def analyze_label_images(label='0'):
    """Analyze all images in a specific label folder to debug hand detection"""
    src_dir = os.path.join("data", "handsigns", label)
    if not os.path.exists(src_dir):
        print(f"Error: Label directory {src_dir} not found!")
        return
    
    images = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nAnalyzing {len(images)} images in label '{label}'...")
    
    # Process each image with visualization
    for img_file in images:
        img_path = os.path.join(src_dir, img_file)
        print(f"\nProcessing {img_file}...")
        landmarks = extract_landmarks_from_image_file(img_path, debug=True)
        if landmarks:
            non_zero = sum(1 for x in landmarks if x != 0.0)
            print(f"Got {non_zero} non-zero landmarks out of 126")
        else:
            print("No landmarks detected!")

if __name__ == '__main__':
    # Change '0' to whatever label you want to analyze
    analyze_label_images('0')