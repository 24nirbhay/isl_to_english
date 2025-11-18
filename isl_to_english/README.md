# ISL to English Translator

This project converts Indian Sign Language (ISL) gestures to English text using MediaPipe hand landmarks and a sequence model.

Overview
- `app.py`: single CLI entrypoint with commands: `prepare`, `train`, `run`, `collect`, `gpu-check`.
- `src/data_collection.py`: utilities to collect webcam sequences and convert images to landmark CSVs.
- `src/prepare_dataset.py`: helper script to convert `data/handsigns/` images → `data/dataset/` CSVs.
- `src/preprocess.py`: loads CSV sequences and prepares arrays for model training.
- `src/model_architecture.py`: model definition (bidirectional LSTM).
- `src/model_train.py`: training pipeline — loads dataset, trains, saves artifacts.
- `src/real_time_translator.py`: live translator with sentence accumulation and pause timer.
- `src/gpu_check.py`: quick TensorFlow GPU availability test.

Quickstart (Windows PowerShell)

1. Create and activate a virtual environment
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Verify GPU (optional)
```powershell
python app.py gpu-check
```

3. Prepare dataset from images
```powershell
python app.py prepare
```

4. Train the model
```powershell
python app.py train
```

5. Run the real-time translator
```powershell
python app.py run
```

Troubleshooting
- If MediaPipe does not detect hands in some images, run `python src/debug_detection.py` to generate debug images at `data/debug_detection/`.
- If you face protobuf/TensorFlow incompatibilities, ensure `protobuf==4.25.3` with Mediapipe 0.10.20 and TensorFlow 2.17.

License: MIT
