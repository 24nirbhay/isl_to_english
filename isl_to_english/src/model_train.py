import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
from .preprocess import load_dataset

def train_model():
    X, y = load_dataset()
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    dump(le, "models/tokenizer.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0., input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.LSTM(128, return_sequences=True, activation='relu'),
        tf.keras.layers.LSTM(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(set(y_encoded)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    model.save("models/isl_seq2seq_model.h5")
