"""
Deep Learning Project 2: ECG Multi-Class Classification (MIT-BIH Dataset)
Author: Moeinreza Bagheri
Description:
    This script processes the MIT-BIH Arrhythmia Dataset.
    It classifies heartbeats into 5 categories:
    0: Normal
    1: Supraventricular premature beat
    2: Premature ventricular contraction
    3: Fusion of ventricular and normal beat
    4: Unclassifiable beat
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
TRAIN_PATH = 'data/mitbih_train.csv'
TEST_PATH = 'data/mitbih_test.csv'
OUTPUT_DIR = 'outputs'

BATCH_SIZE = 128
EPOCHS = 10
NUM_CLASSES = 5

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- GPU Configuration ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" GPU Detected: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(e)
else:
    print(" No GPU detected. Running on CPU.")


def load_mitbih_data():
    """
    Loads pre-split train and test datasets from CSV files.
    Reshapes the input for RNN/GRU compatibility.
    """
    print(" Loading MIT-BIH dataset...")

    df_train = pd.read_csv(TRAIN_PATH, header=None)
    df_test = pd.read_csv(TEST_PATH, header=None)

    # Separate Features and Labels
    # Last column is the class ID (0-4)
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values.astype(int)

    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values.astype(int)

    # Reshape for GRU (Samples, TimeSteps, 1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(f" Data Loaded. Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def build_multi_class_model(input_shape, num_classes):
    """
    Builds a GRU model optimized for multi-class classification.
    ** Optimized for GPU speed by removing Masking layer **
    """
    model = models.Sequential([

        layers.GRU(64, return_sequences=True, dropout=0.2, input_shape=input_shape),
        layers.GRU(32, dropout=0.2),

        layers.Dense(32, activation='relu'),

        # Output layer: 5 neurons for 5 classes (Softmax)
        layers.Dense(num_classes, activation='softmax')
    ])

    # Loss function for integer labels: sparse_categorical_crossentropy
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_and_save_results(history, y_true, y_pred):
    """
    Visualizes training performance and confusion matrix.
    """
    # 1. Plot Training History
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('MIT-BIH Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('MIT-BIH Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, 'mitbih_history.png'))
    plt.show()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix (MIT-BIH)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, 'mitbih_confusion_matrix.png'))
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    X_train, y_train, X_test, y_test = load_mitbih_data()

    # 2. Build Model
    model = build_multi_class_model(input_shape=(X_train.shape[1], 1), num_classes=NUM_CLASSES)
    model.summary()

    # 3. Train Model
    print("\n Starting Training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,  # Increased batch size
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 4. Save Model
    model_save_path = os.path.join(OUTPUT_DIR, 'mitbih_model.h5')
    model.save(model_save_path)
    print(f"\n Model saved to {model_save_path}")

    # 5. Evaluate
    print("\n Evaluating on Test Set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Get the class with highest probability

    plot_and_save_results(history, y_test, y_pred)

    print("\n--- Classification Report ---")
    class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
    print(classification_report(y_test, y_pred, target_names=class_names))