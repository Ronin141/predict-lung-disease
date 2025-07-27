# predict.py - Single Audio File Prediction Script

import numpy as np
import librosa
import tensorflow as tf
import pickle
import os
import argparse

# Label mapping
LABEL_MAPPING = {
    0: "Normal",
    1: "Crackles", 
    2: "Wheezes",
    3: "Both Crackles and Wheezes"
}

def load_model_and_encoder(model_path='lung_sound_model.h5', encoder_path='label_encoder.pkl'):
    """
    Load trained model and label encoder
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Label encoder loaded successfully!")
        
        return model, label_encoder
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def preprocess_audio(audio_file_path):
    """
    Preprocess audio file for prediction
    """
    try:
        print(f"Processing audio file: {audio_file_path}")
        
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=4000, duration=10)
        print(f"Audio loaded - Duration: {len(y)/sr:.2f}s, Sample rate: {sr}Hz")
        
        # Convert to mel-spectrogram  
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, n_fft=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Reshape for model input
        mel_spec_db = mel_spec_db.reshape(1, mel_spec_db.shape[0], mel_spec_db.shape[1], 1)
        print(f"Processed shape: {mel_spec_db.shape}")
        
        return mel_spec_db
        
    except Exception as e:
        print(f"❌ Error preprocessing audio: {e}")
        return None

def predict_lung_sound(model, label_encoder, audio_file_path):
    """
    Predict lung sound classification
    """
    # Preprocess audio
    processed_audio = preprocess_audio(audio_file_path)
    if processed_audio is None:
        return None
    
    print("Making prediction...")
    
    # Make prediction
    prediction = model.predict(processed_audio, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))
    
    # Get label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    readable_label = LABEL_MAPPING.get(predicted_label, f"Unknown ({predicted_label})")
    
    return {
        "predicted_class": int(predicted_class),
        "predicted_label": int(predicted_label),
        "readable_label": readable_label,
        "confidence": confidence,
        "all_probabilities": prediction[0].tolist()
    }

def print_results(result):
    """
    Print prediction results in a nice format
    """
    if result is None:
        print("❌ Prediction failed!")
        return
    
    print("\n" + "="*50)
    print("🔍 LUNG SOUND PREDICTION RESULTS")
    print("="*50)
    print(f"📊 Prediction: {result['readable_label']}")
    print(f"🎯 Confidence: {result['confidence']:.2%}")
    print(f"🔢 Class ID: {result['predicted_label']}")
    
    print(f"\n📈 All Class Probabilities:")
    for i, prob in enumerate(result['all_probabilities']):
        label = LABEL_MAPPING.get(i, f"Class {i}")
        print(f"   {label}: {prob:.4f} ({prob*100:.2f}%)")
    
    print("="*50)
    
    # Interpretation
    if result['confidence'] > 0.8:
        print("✅ High confidence prediction")
    elif result['confidence'] > 0.6:
        print("⚠️  Medium confidence prediction")
    else:
        print("🔄 Low confidence prediction - consider getting more samples")

def main():
    parser = argparse.ArgumentParser(description='Predict lung sound from audio file')
    parser.add_argument('audio_file', type=str, help='Path to audio file')
    parser.add_argument('--model', type=str, default='lung_sound_model.h5', 
                       help='Path to model file (default: lung_sound_model.h5)')
    parser.add_argument('--encoder', type=str, default='label_encoder.pkl',
                       help='Path to encoder file (default: label_encoder.pkl)')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        return
    
    # Load model and encoder
    print("Loading model and encoder...")
    model, label_encoder = load_model_and_encoder(args.model, args.encoder)
    
    if model is None or label_encoder is None:
        print("❌ Failed to load model. Please train the model first by running:")
        print("   python train_model.py")
        return
    
    # Make prediction
    result = predict_lung_sound(model, label_encoder, args.audio_file)
    
    # Print results
    print_results(result)

if __name__ == "__main__":
    main()