# debug_model.py - Script to debug binary lung sound classification issues

import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

def analyze_model_and_data():
    """
    Analyze binary classification model predictions and structure
    """
    print("=== Model Analysis ===")

    # Load model
    try:
        model = tf.keras.models.load_model('lung_sound_binary_model.h5')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # 1. Check model architecture
    print("\n1. Model Architecture:")
    model.summary()

    # 2. Test with dummy data
    print("\n2. Testing with dummy data:")
    dummy_input = np.random.randn(1, 64, 128, 1)  # Adjust shape to your model
    dummy_prediction = model.predict(dummy_input, verbose=0)
    print(f"   Dummy prediction: {dummy_prediction[0][0]:.4f}")
    predicted_class = int(dummy_prediction[0][0] > 0.5)
    print(f"   Predicted class: {predicted_class} ({'Abnormal' if predicted_class else 'Normal'})")

    # 3. Test with multiple random inputs
    print("\n3. Testing multiple random inputs:")
    predictions = []
    for i in range(10):
        random_input = np.random.randn(1, 64, 128, 1)
        pred = model.predict(random_input, verbose=0)
        conf = float(pred[0][0])
        predicted_class = int(conf > 0.5)
        predictions.append(predicted_class)
        print(f"   Test {i+1}: Predicted class {predicted_class} ({'Abnormal' if predicted_class else 'Normal'}), confidence: {conf:.3f}")

    prediction_counts = Counter(predictions)
    print(f"\n   Prediction distribution: {dict(prediction_counts)}")

    if len(set(predictions)) == 1:
        print("⚠️  WARNING: Model always predicts the same class! Possible training issue.")

    return model

def analyze_audio_preprocessing(audio_file_path):
    """
    Analyze audio preprocessing step by step
    """
    print(f"\n=== Audio Preprocessing Analysis: {audio_file_path} ===")

    try:
        y, sr = librosa.load(audio_file_path, sr=4000, duration=10)
        print(f"1. Audio loaded: length={len(y)}, sr={sr}, duration={len(y)/sr:.2f}s")
        print(f"   Audio range: [{np.min(y):.4f}, {np.max(y):.4f}]")
        print(f"   Audio mean: {np.mean(y):.4f}, std: {np.std(y):.4f}")

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, n_fft=512, hop_length=128
        )
        print(f"2. Mel-spectrogram shape: {mel_spec.shape}")

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        print(f"3. Mel-spectrogram dB range: [{np.min(mel_spec_db):.2f}, {np.max(mel_spec_db):.2f}]")

        current_length = mel_spec_db.shape[1]
        if current_length < 128:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 128 - current_length)), 
                               mode='constant', constant_values=mel_spec_db.min())
            print(f"4. Padded from {current_length} to 128")
        elif current_length > 128:
            mel_spec_db = mel_spec_db[:, :128]
            print(f"4. Truncated from {current_length} to 128")

        print(f"5. Final shape: {mel_spec_db.shape}")

        mel_spec_db = mel_spec_db.reshape(1, 64, 128, 1)
        print(f"6. Reshaped for model: {mel_spec_db.shape}")

        return mel_spec_db

    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return None

def test_multiple_files(file_paths, model):
    """
    Test binary model on multiple audio files
    """
    print("\n=== Testing Multiple Files ===")

    all_predictions = []

    for i, file_path in enumerate(file_paths):
        try:
            processed_audio = analyze_audio_preprocessing(file_path)
            if processed_audio is not None:
                prediction = model.predict(processed_audio, verbose=0)
                confidence = float(prediction[0][0])
                predicted_class = int(confidence > 0.5)
                readable_label = "Abnormal" if predicted_class == 1 else "Normal"

                all_predictions.append(predicted_class)

                print(f"\nFile {i+1} ({file_path}):")
                print(f"   Predicted class: {predicted_class}")
                print(f"   Readable label: {readable_label}")
                print(f"   Confidence: {confidence:.3f}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    prediction_counts = Counter(all_predictions)
    print(f"\n=== Overall Prediction Summary ===")
    print(f"Prediction distribution: {dict(prediction_counts)}")

    if len(set(all_predictions)) == 1:
        print("⚠️  CRITICAL: All files predicted as same class! Likely bias.")

    return all_predictions

def suggest_fixes():
    """
    Suggest fixes for binary classification model issues
    """
    print("\n=== Suggested Fixes ===")
    print("""
1. **Check Data Balance:**
   - Ensure both classes (0/1) have similar number of samples
   - Use `class_weight` in model.fit() if imbalance exists

2. **Preprocessing Consistency:**
   - Confirm that training and inference preprocess the audio identically

3. **Metrics Monitoring:**
   - Use accuracy, precision, recall, and F1-score during validation

4. **Training Monitoring:**
   - Plot confusion matrix
   - Use early stopping with validation loss
   - Track per-class performance

5. **Model Architecture:**
   - Try different architectures
   - Add dropout/batch normalization
   - Try using focal loss for class imbalance

6. **Data Quality:**
   - Ensure audio samples are labeled correctly
   - Perform data augmentation for minority class if needed
    """)

def check_training_data_balance(data_dir):
    """
    Check class distribution in the dataset directory
    """
    import os
    from collections import defaultdict

    print("\n=== Training Data Balance Check ===")

    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return

    class_counts = defaultdict(int)

    for root, dirs, files in os.walk(data_dir):
        class_name = os.path.basename(root)
        audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]

        if audio_files:
            class_counts[class_name] += len(audio_files)

    print("Class distribution in training data:")
    total = sum(class_counts.values())
    for cls, count in class_counts.items():
        percent = (count / total) * 100 if total > 0 else 0
        print(f"   {cls}: {count} samples ({percent:.1f}%)")

    if class_counts:
        max_c = max(class_counts.values())
        min_c = min(class_counts.values())
        ratio = max_c / min_c if min_c > 0 else float('inf')

        if ratio > 5:
            print(f"⚠️  SEVERE imbalance: {ratio:.1f}:1")
        elif ratio > 2:
            print(f"⚠️  Moderate imbalance: {ratio:.1f}:1")

if __name__ == "__main__":
    print("🔍 Lung Sound Model Diagnostic Tool (Binary Classification)")
    print("="*60)

    # 1. Analyze model
    model = analyze_model_and_data()

    # 2. Optionally check training data
    # check_training_data_balance("path/to/training/data")

    # 3. Test with real files (add your file paths here)
    sample_files = [
        # "samples/audio1.wav",
        # "samples/audio2.wav"
    ]

    if sample_files and model is not None:
        test_multiple_files(sample_files, model)
    else:
        print("\n⚠️  No test files provided. Please add audio paths to sample_files list.")

    # 4. Suggest fixes
    suggest_fixes()

    print("\n" + "="*60)
    print("🏁 Diagnostic complete!")
