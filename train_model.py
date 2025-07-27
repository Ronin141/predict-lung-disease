import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization, MaxPooling2D, 
                                     Dropout, GlobalAveragePooling2D, Dense)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle

def load_icbhi_data(data_path, sr=4000, n_mels=64, target_len=128):
    X, y = [], []

    for fname in os.listdir(data_path):
        if fname.endswith('.txt'):
            basename = fname.replace('.txt', '')
            wav_path = os.path.join(data_path, basename + '.wav')
            txt_path = os.path.join(data_path, fname)

            if os.path.exists(wav_path):
                try:
                    y_audio, sr = librosa.load(wav_path, sr=sr)
                    with open(txt_path) as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 4:
                                start, end = map(float, parts[:2])
                                crackles, wheezes = map(int, parts[2:4])

                                if crackles == 0 and wheezes == 0: label = 0
                                elif crackles == 1 and wheezes == 0: label = 1
                                elif crackles == 0 and wheezes == 1: label = 2
                                else: label = 3

                                segment = y_audio[int(start*sr):int(end*sr)]
                                if len(segment) >= sr:
                                    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
                                    mel_db = librosa.power_to_db(mel, ref=np.max)
                                    padded = pad_or_truncate(mel_db, target_len)
                                    normalized = librosa.util.normalize(padded)
                                    X.append(normalized)
                                    y.append(label)
                except Exception as e:
                    print(f"Error processing {wav_path}: {e}")
    return np.array(X), np.array(y)

def pad_or_truncate(mel_spec, target_len):
    if mel_spec.shape[1] < target_len:
        pad_width = target_len - mel_spec.shape[1]
        return np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return mel_spec[:, :target_len]

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.4),

        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_history(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.title('Accuracy Over Epochs')
    plt.tight_layout()
    plt.savefig('acc_curve.png')
    plt.show()

def main():
    DATA_PATH = r'E:\My work\predict-lung-disease\data\ICBHI_final_database'
    MODEL_OUT = 'lung_model_v2.h5'
    ENCODER_OUT = 'lung_labels_v2.pkl'

    X_raw, y_raw = load_icbhi_data(DATA_PATH)
    print(f"Loaded: {X_raw.shape}")

    X = X_raw[..., np.newaxis]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    y_cat = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, stratify=y_cat, random_state=42)

    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    model = build_model(X.shape[1:], num_classes=y_cat.shape[1])
    model.summary()

    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=100, batch_size=32,
        callbacks=callbacks, class_weight=class_weights_dict, verbose=1
    )

    plot_history(history)
    model.save(MODEL_OUT)
    with open(ENCODER_OUT, 'wb') as f:
        pickle.dump(encoder, f)

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_.astype(str)))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig('confusion_matrix_v2.png')
    plt.show()

if __name__ == '__main__':
    main()