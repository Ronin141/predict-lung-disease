# app.py - FastAPI Server for Binary Lung Sound Classification

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
from load_model import download_model # Import the model loading function

# Initialize FastAPI app
app = FastAPI(
    title="Binary Lung Sound Classification API",
    description="API for binary classification of lung sounds (Normal vs Abnormal) with percentage output",
    version="2.0.0"
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
normalization_params = None
MODEL_ID = "1ZLVYUb0PHTSAk6HlcGOjFa691iYLh4f5"
MODEL_PATH = 'lung_sound_binary_model.h5'
NORM_PARAMS_PATH = 'binary_label_encoder.pkl'

# Constants for preprocessing
TARGET_LENGTH = 128  # Fixed length for mel-spectrogram
N_MELS = 64         # Number of mel bands
SAMPLE_RATE = 4000  # Sample rate
DURATION = 10       # Duration in seconds

def load_model_and_params():
    """
    Load trained binary model and normalization parameters
    """
    global model, normalization_params
    download_model()
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found: {MODEL_PATH}")
            return False
            
        if not os.path.exists(NORM_PARAMS_PATH):
            print(f"Normalization parameters file not found: {NORM_PARAMS_PATH}")
            print("Will use default normalization")
        
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Binary model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Load normalization parameters
        try:
            with open(NORM_PARAMS_PATH, 'rb') as f:
                normalization_params = pickle.load(f)
            print("Normalization parameters loaded successfully!")
            print(f"X_min: {normalization_params['X_min']:.4f}, X_max: {normalization_params['X_max']:.4f}")
        except Exception as e:
            print(f"Warning: Could not load normalization parameters: {e}")
            normalization_params = None
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        normalization_params = None
        return False

def pad_or_truncate(mel_spec, target_length=128):
    """Pad or truncate mel-spectrogram to target length"""
    if mel_spec.shape[1] > target_length:
        return mel_spec[:, :target_length]
    else:
        pad_width = target_length - mel_spec.shape[1]
        return np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')

def preprocess_audio(audio_file_path):
    """
    Preprocess audio file for binary prediction
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
            n_fft=512
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        print(f"Mel-spectrogram shape before padding/truncating: {mel_spec_db.shape}")
        
        # Pad or truncate to target length
        mel_spec_db = pad_or_truncate(mel_spec_db, TARGET_LENGTH)
        print(f"Final mel-spectrogram shape: {mel_spec_db.shape}")
        
        # Reshape for model input: (batch, height, width, channels)
        mel_spec_db = mel_spec_db.reshape(1, mel_spec_db.shape[0], mel_spec_db.shape[1], 1)
        mel_spec_db = mel_spec_db.astype(np.float32)
        print(f"Reshaped for model: {mel_spec_db.shape}")
        
        # Apply normalization if parameters are available
        if normalization_params is not None:
            X_min = normalization_params['X_min']
            X_max = normalization_params['X_max']
            if X_max > X_min:
                mel_spec_db = (mel_spec_db - X_min) / (X_max - X_min)
                print("Applied saved normalization parameters")
        else:
            # Default normalization
            X_min, X_max = mel_spec_db.min(), mel_spec_db.max()
            if X_max > X_min:
                mel_spec_db = (mel_spec_db - X_min) / (X_max - X_min)
                print("Applied default normalization")
        
        # Ensure the shape matches exactly what the model expects
        expected_shape = (1, N_MELS, TARGET_LENGTH, 1)
        if mel_spec_db.shape != expected_shape:
            raise ValueError(f"Shape mismatch! Expected {expected_shape}, got {mel_spec_db.shape}")
        
        return mel_spec_db
        
    except Exception as e:
        raise Exception(f"Error preprocessing audio: {str(e)}")

def predict_lung_sound_percentage(audio_file_path):
    """
    Predict lung sound normalcy in 5 steps: 0, 25, 50, 75, 100
    """
    if model is None:
        raise ValueError("Model not loaded")
    
    try:
        processed_audio = preprocess_audio(audio_file_path)
        prediction = model.predict(processed_audio, verbose=0)
        probability = float(prediction[0][0])  # between 0 and 1

        # Map to 5 steps: 0, 25, 50, 75, 100
        if probability < 0.1:
            step_percentage = 0
        elif probability < 0.35:
            step_percentage = 25
        elif probability < 0.65:
            step_percentage = 50
        elif probability < 0.9:
            step_percentage = 75
        else:
            step_percentage = 100

        predicted_class = 1 if probability > 0.5 else 0
        confidence = probability if predicted_class == 1 else (1 - probability)

        # Determine risk level
        if step_percentage == 100:
            status = "Normal"
            status_detail = "ปกติ"
            risk_level = "Low"
        elif step_percentage == 75:
            status = "Mildly Normal"
            status_detail = "เกือบปกติ"
            risk_level = "Low-Medium"
        elif step_percentage == 50:
            status = "Borderline"
            status_detail = "ผิดปกติเล็กน้อย"
            risk_level = "Medium"
        elif step_percentage == 25:
            status = "Abnormal"
            status_detail = "ผิดปกติ"
            risk_level = "High"
        else:
            status = "Highly Abnormal"
            status_detail = "ผิดปกติมาก"
            risk_level = "Critical"

        return {
            "percentage": step_percentage,
            "probability": round(probability, 4),
            "predicted_class": int(predicted_class),
            "classification": status,
            "status_thai": status_detail,
            "risk_level": risk_level,
            "confidence": round(confidence * 100, 1),
            "interpretation": {
                "0": "ผิดปกติมาก (Highly Abnormal)",
                "25": "ผิดปกติ (Abnormal)",
                "50": "ผิดปกติเล็กน้อย (Borderline)",
                "75": "เกือบปกติ (Mildly Normal)",
                "100": "ปกติ (Normal)"
            }
        }
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


# Event handlers - Using lifespan instead of deprecated on_event
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Binary Lung Sound Classification API...")
    success = load_model_and_params()
    
    if not success:
        print("⚠️  WARNING: Model not loaded!")
        print("Please train the binary model first by running the training script")
        print("The API will return errors until a model is available.")
    else:
        print("✅ Binary model loaded successfully! API is ready.")
        print("📊 Output format: 0% = Abnormal, 100% = Normal")
    
    yield
    
    # Shutdown
    print("API is shutting down...")

# Update app initialization with lifespan
app = FastAPI(
    title="Binary Lung Sound Classification API",
    description="API for binary classification of lung sounds (Normal vs Abnormal) with percentage output",
    version="2.0.0",
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
        "message": "Binary Lung Sound Classification API",
        "status": "running",
        "classification_type": "Binary (Normal vs Abnormal)",
        "output_format": "0% = Abnormal, 100% = Normal",
        "model_loaded": model is not None,
        "expected_input_shape": f"({N_MELS}, {TARGET_LENGTH}, 1)",
        "endpoints": {
            "predict": "/predict/ - POST audio file for percentage classification",
            "health": "/health - GET health status",
            "info": "/info - GET model information",
            "test": "/test-prediction - GET example prediction format"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_type": "Binary Classification",
        "model_loaded": model is not None,
        "normalization_loaded": normalization_params is not None,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "norm_params_exists": os.path.exists(NORM_PARAMS_PATH),
        "expected_shape": f"(1, {N_MELS}, {TARGET_LENGTH}, 1)",
        "output_format": "Percentage (0% = Abnormal, 100% = Normal)"
    }

@app.get("/info")
async def model_info():
    """
    Get model information
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Binary model not loaded. Please train the model first."
        )
    
    return {
        "model_type": "Binary CNN for Lung Sound Classification",
        "classification": "Binary (Normal vs Abnormal)",
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "output_format": {
            "percentage": "0-100% (0% = Abnormal, 100% = Normal)",
            "interpretation": {
                "0-30%": "ผิดปกติมาก (Highly Abnormal)",
                "30-70%": "ผิดปกติเล็กน้อย (Borderline)", 
                "70-100%": "ปกติ (Normal)"
            }
        },
        "expected_preprocessing": {
            "sample_rate": SAMPLE_RATE,
            "duration": DURATION,
            "n_mels": N_MELS,
            "target_length": TARGET_LENGTH
        },
        "normalization_available": normalization_params is not None
    }

@app.get("/test-prediction")
async def test_prediction_format():
    """
    Show example prediction format
    """
    return {
        "example_response": {
            "status": "success",
            "filename": "example_lung_sound.wav",
            "prediction": {
                "percentage": 85.3,
                "probability": 0.853,
                "predicted_class": 1,
                "classification": "Normal",
                "status_thai": "ปกติ",
                "risk_level": "Low",
                "confidence": 85.3,
                "interpretation": {
                    "0-30%": "ผิดปกติมาก (Highly Abnormal)",
                    "30-70%": "ผิดปกติเล็กน้อย (Borderline)",
                    "70-100%": "ปกติ (Normal)"
                }
            },
            "message": "Lung sound is 85.3% normal (ปกติ)",
            "processing_info": {
                "sample_rate": 4000,
                "duration": 10,
                "n_mels": 64,
                "target_length": 128
            }
        }
    }

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict lung sound normalcy percentage from uploaded audio file
    Returns percentage: 0% = Abnormal, 100% = Normal
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Binary model not loaded. Please train the model first."
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
        result = predict_lung_sound_percentage(tmp_file_path)
        
        # Create response message
        percentage = result['percentage']
        status_thai = result['status_thai']
        message = f"เสียงปอดมีความปกติ {percentage}% ({status_thai})"
        
        # Return result
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "prediction": result,
            "message": message,
            "processing_info": {
                "sample_rate": SAMPLE_RATE,
                "duration": DURATION,
                "n_mels": N_MELS,
                "target_length": TARGET_LENGTH,
                "model_type": "Binary Classification"
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

@app.post("/predict-batch/")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict multiple audio files at once
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Binary model not loaded. Please train the model first."
        )
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Too many files. Maximum 10 files per batch."
        )
    
    results = []
    
    for file in files:
        # Validate file type
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": "Invalid file type"
            })
            continue
        
        tmp_file_path = None
        
        try:
            # Save uploaded file temporarily
            content = await file.read()
            if len(content) > 50 * 1024 * 1024:
                results.append({
                    "filename": file.filename,
                    "status": "error", 
                    "error": "File too large"
                })
                continue
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            # Make prediction
            prediction = predict_lung_sound_percentage(tmp_file_path)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "prediction": prediction,
                "message": f"เสียงปอดมีความปกติ {prediction['percentage']}% ({prediction['status_thai']})"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
        
        finally:
            # Clean up temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
    
    return JSONResponse(content={
        "status": "completed",
        "total_files": len(files),
        "results": results
    })

@app.post("/reload-model/")
async def reload_model():
    """
    Reload the model (useful after retraining)
    """
    try:
        success = load_model_and_params()
        
        if success:
            return JSONResponse(content={
                "status": "success",
                "message": "Binary model reloaded successfully",
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape),
                "model_type": "Binary Classification"
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
            "available_endpoints": [
                "/", "/health", "/info", "/predict/", 
                "/predict-batch/", "/test-prediction", "/reload-model/"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error occurred",
            "model_type": "Binary Classification"
        }
    )

if __name__ == "__main__":
    print("Starting Binary Lung Sound Classification API Server...")
    print("Make sure you have trained the binary model first!")
    print(f"\nModel Configuration:")
    print(f"- Classification: Binary (Normal vs Abnormal)")
    print(f"- Output: Percentage (0% = Abnormal, 100% = Normal)")
    print(f"- Sample rate: {SAMPLE_RATE} Hz")
    print(f"- Duration: {DURATION} seconds")
    print(f"- Mel bands: {N_MELS}")
    print(f"- Target length: {TARGET_LENGTH}")
    print(f"- Expected input shape: (1, {N_MELS}, {TARGET_LENGTH}, 1)")
    print(f"\nInterpretation:")
    print(f"- 0-30%: ผิดปกติมาก (Highly Abnormal)")
    print(f"- 30-70%: ผิดปกติเล็กน้อย (Borderline)")
    print(f"- 70-100%: ปกติ (Normal)")
    print("\nAPI will be available at:")
    print("- http://localhost:8000 (main endpoint)")
    print("- http://localhost:8000/docs (API documentation)")
    print("- http://localhost:8000/predict/ (prediction endpoint)")
    print("- http://localhost:8000/test-prediction (example format)")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)