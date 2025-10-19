import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from joblib import dump

def train_model():
    """
    Trains an image classification model for hand gestures using transfer learning.
    """
    # --- 1. Configuration ---
    # Assumes your dataset is in 'project_root/data/images'
    DATA_DIR = os.path.join('..', 'data', 'images') 

    # Model-specific parameters
    IMG_SIZE = (224, 224) # Input size for MobileNetV2
    BATCH_SIZE = 32       # Number of images to process in a single batch

    # --- 2. Create Datasets (Training and Validation) ---
    print("Creating training dataset...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,  # Use 20% of the data for validation
        subset="training",
        seed=42,               # Seed for reproducibility
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    print("Creating validation dataset...")
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # --- 3. Extract and Save Class Names ---
    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"\nFound {num_classes} classes. Saving class names...")
    # Save the class names for use during inference
    dump(class_names, os.path.join('..', 'models', 'image_class_names.joblib'))

    # --- 4. Configure Dataset for Performance ---
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print("\nData pipeline is ready!")

    # --- 5. Define Data Augmentation and Preprocessing Layers ---
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # --- 6. Create the Model using Transfer Learning ---
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    # --- 7. Chain the Layers Together ---
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    # --- 8. Compile the Model ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    model.summary()

    # --- 9. Train the Model ---
    EPOCHS = 10
    print("\nStarting model training...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )
    print("\nTraining finished!")

    # --- 10. Save the Trained Model ---
    model.save(os.path.join('..', 'models', 'image_gesture_recognizer.keras'))
    print("\nModel saved to models/image_gesture_recognizer.keras")

    # --- 11. Visualize Performance ---
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    if not os.path.exists(os.path.join('..', 'models')):
        os.makedirs(os.path.join('..', 'models'))
    train_model()