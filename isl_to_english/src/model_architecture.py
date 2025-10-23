import tensorflow as tf
from tensorflow.keras import layers

def create_sign_language_model(time_steps, feature_dim, num_classes):
    """Create a more sophisticated model architecture for sign language recognition.
    
    Args:
        time_steps: Number of time steps in the sequence
        feature_dim: Number of features per time step (126 for hand landmarks)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Input and masking
    inputs = layers.Input(shape=(time_steps, feature_dim))
    x = layers.Masking(mask_value=0.)(inputs)
    
    # Batch normalization for input
    x = layers.BatchNormalization()(x)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    
    # Dense layers with residual connections
    dense1 = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(dense1)
    x = layers.BatchNormalization()(x)
    
    dense2 = layers.Dense(128, activation='relu')(x)
    x = layers.Concatenate()([dense1, dense2])  # Skip connection
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Compile with learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    return model