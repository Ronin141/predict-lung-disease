# app.py - FastAPI Server for Lung Sound Classification

import numpy as np
import librosa
import tensorflow as tf
import pickle
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Lung Sound Classification API",
    description="API for classifying lung sounds using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
label_encoder = None
MODEL_PATH = 'lung_sound_model.h5'
ENCODER_PATH = 'label_encoder.pkl'

# Constants for preprocessing
TARGET_LENGTH = 128  # Fixed length for mel-spectrogram
N_MELS = 64         # Number of mel bands
SAMPLE_RATE = 4000  # Sample rate
DURATION = 10       # Duration in seconds

# Label mapping for display
LABEL_MAPPING = {
    0: "Normal",
    1: "Crackles", 
    2: "Wheezes",
    3: "Both Crackles and Wheezes"
}

def load_model_and_encoder():
    """
    Load trained model and label encoder
    """
    global model, label_encoder
    
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found: {MODEL_PATH}")
            return False
            
        if not os.path.exists(ENCODER_PATH):
            print(f"Encoder file not found: {ENCODER_PATH}")
            return False
        
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        
        # Load label encoder
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Label encoder loaded successfully!")
        print(f"Classes: {label_encoder.classes_}")
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        label_encoder = None
        return False

def preprocess_audio(audio_file_path):
    """
    Preprocess audio file for prediction with fixed dimensions
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE, duration=DURATION)
        print(f"Loaded audio: length={len(y)}, sr={sr}")
        
        # Convert to mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=N_MELS, 
            n_fft=512,
            hop_length=128
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        print(f"Mel-spectrogram shape before fixing: {mel_spec_db.shape}")
        
        # Fix the time dimension to TARGET_LENGTH
        current_length = mel_spec_db.shape[1]
        
        if current_length < TARGET_LENGTH:
            # Pad if too short
            pad_width = TARGET_LENGTH - current_length
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=mel_spec_db.min())
            print(f"Padded from {current_length} to {TARGET_LENGTH}")
        elif current_length > TARGET_LENGTH:
            # Truncate if too long
            mel_spec_db = mel_spec_db[:, :TARGET_LENGTH]
            print(f"Truncated from {current_length} to {TARGET_LENGTH}")
        
        print(f"Final mel-spectrogram shape: {mel_spec_db.shape}")
        
        # Reshape for model input: (batch, height, width, channels)
        mel_spec_db = mel_spec_db.reshape(1, mel_spec_db.shape[0], mel_spec_db.shape[1], 1)
        print(f"Reshaped for model: {mel_spec_db.shape}")
        
        # Ensure the shape matches exactly what the model expects
        expected_shape = (1, N_MELS, TARGET_LENGTH, 1)
        if mel_spec_db.shape != expected_shape:
            raise ValueError(f"Shape mismatch! Expected {expected_shape}, got {mel_spec_db.shape}")
        
        return mel_spec_db
        
    except Exception as e:
        raise Exception(f"Error preprocessing audio: {str(e)}")

def predict_lung_sound(audio_file_path):
    """
    Predict lung sound classification
    """
    if model is None or label_encoder is None:
        raise ValueError("Model not loaded")
    
    try:
        # Preprocess audio
        processed_audio = preprocess_audio(audio_file_path)
        print(f"Input shape for prediction: {processed_audio.shape}")
        
        # Make prediction
        prediction = model.predict(processed_audio, verbose=0)
        print(f"Raw prediction: {prediction}")
        
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Get label
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        
        # Get readable label
        readable_label = LABEL_MAPPING.get(predicted_label, f"Unknown ({predicted_label})")
        
        return {
            "predicted_class": int(predicted_class),
            "predicted_label": int(predicted_label),
            "readable_label": readable_label,
            "confidence": confidence,
            "all_probabilities": prediction[0].tolist()
        }
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

# Event handlers - Using lifespan instead of deprecated on_event
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Lung Sound Classification API...")
    success = load_model_and_encoder()
    
    if not success:
        print("⚠️  WARNING: Model not loaded!")
        print("Please train the model first by running: python train_model.py")
        print("The API will return errors until a model is available.")
    else:
        print("✅ Model loaded successfully! API is ready.")
    
    yield
    
    # Shutdown
    print("API is shutting down...")

# Update app initialization with lifespan
app = FastAPI(
    title="Lung Sound Classification API",
    description="API for classifying lung sounds using deep learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Lung Sound Classification API",
        "status": "running",
        "model_loaded": model is not None,
        "expected_input_shape": f"({N_MELS}, {TARGET_LENGTH}, 1)",
        "endpoints": {
            "predict": "/predict/ - POST audio file for classification",
            "health": "/health - GET health status",
            "info": "/info - GET model information"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "encoder_path_exists": os.path.exists(ENCODER_PATH),
        "expected_shape": f"(1, {N_MELS}, {TARGET_LENGTH}, 1)"
    }

@app.get("/info")
async def model_info():
    """
    Get model information
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    return {
        "model_type": "CNN for Lung Sound Classification",
        "input_shape": str(model.input_shape),
        "expected_preprocessing": {
            "sample_rate": SAMPLE_RATE,
            "duration": DURATION,
            "n_mels": N_MELS,
            "target_length": TARGET_LENGTH
        },
        "output_classes": len(label_encoder.classes_),
        "class_labels": label_encoder.classes_.tolist(),
        "readable_labels": list(LABEL_MAPPING.values())
    }

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict lung sound from uploaded audio file
    """
    # Check if model is loaded
    if model is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running train_model.py"
        )
    
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload audio files (.wav, .mp3, .m4a, .flac)"
        )
    
    # Check file size (limit to 50MB)
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 50MB"
        )
    
    tmp_file_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        print(f"Processing file: {file.filename}")
        print(f"Temporary file: {tmp_file_path}")
        
        # Make prediction
        result = predict_lung_sound(tmp_file_path)
        
        # Return result
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "prediction": result,
            "message": f"Predicted: {result['readable_label']} with {result['confidence']:.2%} confidence",
            "processing_info": {
                "sample_rate": SAMPLE_RATE,
                "duration": DURATION,
                "n_mels": N_MELS,
                "target_length": TARGET_LENGTH
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

@app.post("/reload-model/")
async def reload_model():
    """
    Reload the model (useful after retraining)
    """
    try:
        success = load_model_and_encoder()
        
        if success:
            return JSONResponse(content={
                "status": "success",
                "message": "Model reloaded successfully",
                "input_shape": str(model.input_shape)
            })
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to reload model. Check if model files exist."
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading model: {str(e)}"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "message": "Endpoint not found",
            "available_endpoints": ["/", "/health", "/info", "/predict/"]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error occurred"
        }
    )

if __name__ == "__main__":
    print("Starting Lung Sound Classification API Server...")
    print("Make sure you have trained the model first by running: python train_model.py")
    print(f"\nPreprocessing settings:")
    print(f"- Sample rate: {SAMPLE_RATE} Hz")
    print(f"- Duration: {DURATION} seconds")
    print(f"- Mel bands: {N_MELS}")
    print(f"- Target length: {TARGET_LENGTH}")
    print(f"- Expected input shape: (1, {N_MELS}, {TARGET_LENGTH}, 1)")
    print("\nAPI will be available at:")
    print("- http://localhost:8000 (main endpoint)")
    print("- http://localhost:8000/docs (API documentation)")
    print("- http://localhost:8000/predict/ (prediction endpoint)")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)