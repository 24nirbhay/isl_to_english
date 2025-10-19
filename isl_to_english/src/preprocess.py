import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = "data/dataset"

def load_dataset():
    data, labels = [], []
    for gesture in os.listdir(DATA_PATH):
        for file in os.listdir(os.path.join(DATA_PATH, gesture)):
            arr = np.loadtxt(os.path.join(DATA_PATH, gesture, file), delimiter=',', ndmin=2)
            data.append(arr)
            labels.append(gesture)
    
    # Pad sequences to ensure uniform length
    data = pad_sequences(data, padding='post', dtype='float32', maxlen=30)
    
    return np.array(data), np.array(labels)
