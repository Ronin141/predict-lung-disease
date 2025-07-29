# load_model.py
import os
import gdown

MODEL_ID = "1ZLVYUb0PHTSAk6HlcGOjFa691iYLh4f5"  # แทนด้วย Google Drive file ID จริง
MODEL_PATH = "lung_sound_binary_model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)

# download_model()
