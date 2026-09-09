"""
Deep Learning Project 1: ECG Binary Classification (PTBDB Dataset)
Author: Moeinreza Bagheri
Description:
    This script loads the PTBDB dataset (Normal vs. Abnormal), processes the ECG signals,
    and trains a GRU-based Recurrent Neural Network to classify the heartbeats.
    It handles variable-length sequences using masking and padding.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
DATA_PATH_NORMAL = 'data/ptbdb_normal.csv'
DATA_PATH_ABNORMAL = 'data/ptbdb_abnormal.csv'
OUTPUT_DIR = 'outputs'
BATCH_SIZE = 32
EPOCHS = 15

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- GPU Configuration ---
# Helps prevent OOM errors on RTX 3060 by allowing dynamic memory growth
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


def load_and_process_data():
    """
    Loads normal and abnormal CSV files, concatenates them, shuffles,
    and splits into features (X) and labels (y).

    Returns:
        X_train, X_test, y_train, y_test: Split datasets ready for training.
    """
    print(" Loading PTBDB dataset...")

    # Load datasets (Header=None because files contain raw signal data)
    df_normal = pd.read_csv(DATA_PATH_NORMAL, header=None)
    df_abnormal = pd.read_csv(DATA_PATH_ABNORMAL, header=None)

    # Merge datasets
    df = pd.concat([df_normal, df_abnormal], axis=0)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate Features (Signal) and Labels (Last Column)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values

    # Reshape for RNN/GRU input: (Samples, TimeSteps, Features)
    # We have 1 feature (ECG amplitude) per time step.
    X = X.reshape(X.shape[0], X.shape[1], 1)

    print(f" Data Loaded. Input Shape: {X.shape}")

    # Split: 80% Train, 20% Test
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_gru_model(input_shape):
    """
    Builds a GRU-based neural network model.
    Includes a Masking layer to handle zero-padding in signals.
    """
    model = models.Sequential([
        # Masking layer ignores time steps with value 0.0 (padding)
        layers.Masking(mask_value=0.0, input_shape=input_shape),

        # GRU Layers for time-series analysis
        layers.GRU(64, return_sequences=True, dropout=0.2),
        layers.GRU(32, dropout=0.2),

        # Dense layers for classification
        layers.Dense(32, activation='relu'),

        # Output layer: 1 neuron with Sigmoid (Binary Classification)
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_and_save_results(history, y_true, y_pred, model):
    """
    Plots training history and confusion matrix, then saves them to disk.
    """
    # 1. Plot Training History
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, 'ptbdb_history.png'))
    plt.show()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title('Confusion Matrix (PTBDB)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, 'ptbdb_confusion_matrix.png'))
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_and_process_data()

    # 2. Build Model
    model = build_gru_model(input_shape=(X_train.shape[1], 1))
    model.summary()

    # 3. Train Model
    print("\n Starting Training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,  # Use 10% of training data for validation
        verbose=1
    )

    # 4. Save Model
    model_save_path = os.path.join(OUTPUT_DIR, 'ptbdb_model.h5')
    model.save(model_save_path)
    print(f"\n Model saved to {model_save_path}")

    # 5. Evaluate
    print("\n Evaluating on Test Set...")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    plot_and_save_results(history, y_test, y_pred, model)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))