import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = "data/dataset"


def load_dataset(expected_frame_length=126, maxlen=30):
    """Load all CSV sequences from DATA_PATH.

    Each CSV file represents a sequence (N x F) where F should be expected_frame_length.
    If CSVs were saved from single images, they will be 1 x F.
    Returns: X (num_sequences, maxlen, F), y (num_sequences,)
    """
    data, labels = [], []
    for gesture in os.listdir(DATA_PATH):
        gesture_path = os.path.join(DATA_PATH, gesture)
        if not os.path.isdir(gesture_path):
            continue
        for file in os.listdir(gesture_path):
            file_path = os.path.join(gesture_path, file)
            # Ensure numeric read even if rows count is 1
            arr = np.loadtxt(file_path, delimiter=',', ndmin=2)
            # Normalize shape: if vectors are 3*N (old single-hand 63), expand to expected length by padding zeros
            if arr.shape[1] < expected_frame_length:
                pad_width = expected_frame_length - arr.shape[1]
                arr = np.pad(arr, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
            elif arr.shape[1] > expected_frame_length:
                arr = arr[:, :expected_frame_length]

            data.append(arr.astype('float32'))
            labels.append(gesture)

    # Pad sequences (time dimension) to ensure uniform length
    data = pad_sequences(data, padding='post', dtype='float32', maxlen=maxlen)

    return np.array(data), np.array(labels)
