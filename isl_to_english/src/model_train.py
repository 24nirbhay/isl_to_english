import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
import os
import logging
from datetime import datetime
try:
    from .preprocess import load_dataset
    from .model_architecture import create_sign_language_model
except Exception:
    from preprocess import load_dataset
    from model_architecture import create_sign_language_model

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(epochs=50, batch_size=32, validation_split=0.2):
    logging.info("Starting model training process...")
    
    # Create timestamped model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("models", f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and preprocess dataset
    logging.info("Loading dataset...")
    X, y = load_dataset(expected_frame_length=126, maxlen=30)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save label encoder
    dump(le, os.path.join(model_dir, "label_encoder.joblib"))
    with open(os.path.join(model_dir, "classes.txt"), "w") as f:
        f.write("\n".join(le.classes_))
    
    logging.info(f"Found {len(le.classes_)} unique classes: {le.classes_}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=validation_split,
        stratify=y_encoded,
        random_state=42
    )
    
    logging.info(f"Training set shape: {X_train.shape}")
    logging.info(f"Test set shape: {X_test.shape}")
    
    # Create model
    time_steps = X_train.shape[1]
    feature_dim = X_train.shape[2]
    num_classes = len(set(y_encoded))
    
    model = create_sign_language_model(time_steps, feature_dim, num_classes)

    # Setup callbacks
    callbacks = [
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'model.keras'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train model
    logging.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    logging.info("Evaluating model...")
    test_loss, test_acc, test_top3 = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Test accuracy: {test_acc:.4f}")
    logging.info(f"Top-3 accuracy: {test_top3:.4f}")
    
    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(model_dir, 'training_history.csv'), index=False)
    
    # Save model architecture diagram
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(model_dir, 'model_architecture.png'),
        show_shapes=True
    )
    
    logging.info(f"Training complete. Model and artifacts saved to {model_dir}")
    
    return model, history, model_dir


if __name__ == '__main__':
    train_model()
