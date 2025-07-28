# debug_model.py - Script to debug lung sound classification issues

import numpy as np
import librosa
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from collections import Counter

def analyze_model_and_data():
    """
    Analyze model predictions and data distribution
    """
    print("=== Model Analysis ===")
    
    # Load model and encoder
    try:
        model = tf.keras.models.load_model('lung_sound_binary_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Model and encoder loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 1. Check model architecture
    print("\n1. Model Architecture:")
    model.summary()
    
    # 2. Check label encoder
    print(f"\n2. Label Encoder Classes: {label_encoder.classes_}")
    print(f"   Number of classes: {len(label_encoder.classes_)}")
    
    # 3. Test with dummy data
    print("\n3. Testing with dummy data:")
    dummy_input = np.random.randn(1, 64, 128, 1)  # Match your input shape
    dummy_prediction = model.predict(dummy_input, verbose=0)
    print(f"   Dummy prediction shape: {dummy_prediction.shape}")
    print(f"   Dummy prediction: {dummy_prediction[0]}")
    print(f"   Predicted class: {np.argmax(dummy_prediction)}")
    
    # 4. Check if model is always predicting same class
    print("\n4. Testing multiple random inputs:")
    predictions = []
    for i in range(10):
        random_input = np.random.randn(1, 64, 128, 1)
        pred = model.predict(random_input, verbose=0)
        predicted_class = np.argmax(pred)
        predictions.append(predicted_class)
        print(f"   Test {i+1}: Predicted class {predicted_class}, confidence: {np.max(pred):.3f}")
    
    prediction_counts = Counter(predictions)
    print(f"\n   Prediction distribution: {dict(prediction_counts)}")
    
    if len(set(predictions)) == 1:
        print("⚠️  WARNING: Model always predicts the same class!")
        print("   This suggests the model is not properly trained or has severe bias.")
    
    return model, label_encoder

def analyze_audio_preprocessing(audio_file_path):
    """
    Analyze audio preprocessing step by step
    """
    print(f"\n=== Audio Preprocessing Analysis: {audio_file_path} ===")
    
    try:
        # Load audio
        y, sr = librosa.load(audio_file_path, sr=4000, duration=10)
        print(f"1. Audio loaded: length={len(y)}, sr={sr}, duration={len(y)/sr:.2f}s")
        
        # Check audio properties
        print(f"   Audio range: [{np.min(y):.4f}, {np.max(y):.4f}]")
        print(f"   Audio mean: {np.mean(y):.4f}, std: {np.std(y):.4f}")
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, n_fft=512, hop_length=128
        )
        print(f"2. Mel-spectrogram shape: {mel_spec.shape}")
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        print(f"3. Mel-spectrogram dB range: [{np.min(mel_spec_db):.2f}, {np.max(mel_spec_db):.2f}]")
        
        # Fix length
        current_length = mel_spec_db.shape[1]
        if current_length < 128:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 128 - current_length)), 
                               mode='constant', constant_values=mel_spec_db.min())
            print(f"4. Padded from {current_length} to 128")
        elif current_length > 128:
            mel_spec_db = mel_spec_db[:, :128]
            print(f"4. Truncated from {current_length} to 128")
        
        print(f"5. Final shape: {mel_spec_db.shape}")
        
        # Reshape for model
        mel_spec_db = mel_spec_db.reshape(1, 64, 128, 1)
        print(f"6. Reshaped for model: {mel_spec_db.shape}")
        
        return mel_spec_db
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return None

def test_multiple_files(file_paths, model, label_encoder):
    """
    Test model on multiple files to check consistency
    """
    print("\n=== Testing Multiple Files ===")
    
    LABEL_MAPPING = {
        0: "Normal",
        1: "Crackles", 
        2: "Wheezes",
        3: "Both Crackles and Wheezes"
    }
    
    all_predictions = []
    
    for i, file_path in enumerate(file_paths):
        try:
            processed_audio = analyze_audio_preprocessing(file_path)
            if processed_audio is not None:
                prediction = model.predict(processed_audio, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # Get label
                try:
                    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                    readable_label = LABEL_MAPPING.get(predicted_label, f"Unknown ({predicted_label})")
                except:
                    readable_label = f"Class {predicted_class}"
                
                all_predictions.append(predicted_class)
                
                print(f"\nFile {i+1} ({file_path}):")
                print(f"   Predicted class: {predicted_class}")
                print(f"   Readable label: {readable_label}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   All probabilities: {prediction[0]}")
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    # Analyze overall predictions
    prediction_counts = Counter(all_predictions)
    print(f"\n=== Overall Prediction Summary ===")
    print(f"Prediction distribution: {dict(prediction_counts)}")
    
    if len(set(all_predictions)) == 1:
        print("⚠️  CRITICAL: All files predicted as same class!")
        print("This confirms the model has a severe bias issue.")
    
    return all_predictions

def suggest_fixes():
    """
    Suggest potential fixes for the bias issue
    """
    print("\n=== Suggested Fixes ===")
    print("""
1. **Check Training Data Balance:**
   - Count samples per class in your training data
   - Use class_weight parameter in model.fit() if imbalanced
   
2. **Verify Preprocessing Consistency:**
   - Ensure training and inference use identical preprocessing
   - Check normalization/scaling methods
   
3. **Model Training Issues:**
   - Try different loss functions (focal loss for imbalanced data)
   - Use different metrics (precision, recall, F1-score per class)
   - Implement early stopping based on validation loss
   
4. **Debug Training Process:**
   - Plot training/validation accuracy per class
   - Check confusion matrix on validation set
   - Verify label encoding is consistent
   
5. **Model Architecture:**
   - Try different architectures (more/fewer layers)
   - Add dropout for regularization
   - Use different activation functions
   
6. **Data Quality:**
   - Check if audio files are properly labeled
   - Verify all classes have sufficient, diverse samples
   - Consider data augmentation for minority classes
    """)

def check_training_data_balance(data_dir):
    """
    Check balance of training data (if available)
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
            class_counts[class_name] = len(audio_files)
    
    print("Class distribution in training data:")
    total_samples = sum(class_counts.values())
    
    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Check for severe imbalance
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 5:
            print(f"⚠️  SEVERE CLASS IMBALANCE detected! Ratio: {imbalance_ratio:.1f}:1")
            print("   This is likely causing the prediction bias.")
        elif imbalance_ratio > 2:
            print(f"⚠️  Moderate class imbalance detected. Ratio: {imbalance_ratio:.1f}:1")

if __name__ == "__main__":
    print("🔍 Lung Sound Model Diagnostic Tool")
    print("="*50)
    
    # 1. Analyze model
    model, label_encoder = analyze_model_and_data()
    
    # 2. Check training data balance (modify path as needed)
    # check_training_data_balance("path/to/your/training/data")
    
    # 3. Test with sample files (add your audio file paths)
    sample_files = [
        # "path/to/test/file1.wav",
        # "path/to/test/file2.wav",
        # Add more test files here
    ]
    
    if sample_files and model is not None:
        test_multiple_files(sample_files, model, label_encoder)
    else:
        print("\n⚠️  No test files provided. Add audio file paths to sample_files list.")
    
    # 4. Show suggested fixes
    suggest_fixes()
    
    print("\n" + "="*50)
    print("🏁 Diagnostic complete!")