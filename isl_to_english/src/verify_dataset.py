import os
import mediapipe as mp
import cv2
from data_collection import process_image_dataset
import numpy as np

def verify_and_prepare_dataset():
    # 1. Check handsigns folder
    handsigns_dir = os.path.join("data", "handsigns")
    if not os.path.exists(handsigns_dir):
        print(f"Error: {handsigns_dir} directory not found!")
        return
    
    print("\n=== Checking handsigns directory ===")
    total_images = 0
    for label in os.listdir(handsigns_dir):
        label_dir = os.path.join(handsigns_dir, label)
        if not os.path.isdir(label_dir):
            continue
        images = [f for f in os.listdir(label_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Label '{label}': {len(images)} images")
        total_images += len(images)
    
    print(f"\nTotal images found: {total_images}")
    
    # 2. Process images to landmarks if needed
    dataset_dir = os.path.join("data", "dataset")
    print(f"\n=== Processing images to {dataset_dir} ===")
    process_image_dataset("data/handsigns", dataset_dir)
    
    # 3. Verify processed CSVs
    print("\n=== Checking processed CSVs ===")
    total_csvs = 0
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):
            continue
        csvs = [f for f in os.listdir(label_dir) if f.endswith('.csv')]
        print(f"Label '{label}': {len(csvs)} sequences")
        
        # Verify first CSV in each label
        if csvs:
            first_csv = os.path.join(label_dir, csvs[0])
            try:
                data = np.loadtxt(first_csv, delimiter=',', ndmin=2)
                print(f"  Sample shape: {data.shape}")
                if data.shape[1] != 126:
                    print(f"  WARNING: Expected 126 features, got {data.shape[1]}")
            except Exception as e:
                print(f"  ERROR reading {first_csv}: {str(e)}")
        
        total_csvs += len(csvs)
    
    print(f"\nTotal sequences available: {total_csvs}")
    
    if total_csvs == 0:
        print("\nNo sequences found! Please check that:")
        print("1. You have images in data/handsigns/<label>/ folders")
        print("2. Images are .jpg, .jpeg, or .png files")
        print("3. Each label folder contains at least one image")
    else:
        print("\nDataset looks ready for training!")

if __name__ == '__main__':
    verify_and_prepare_dataset()