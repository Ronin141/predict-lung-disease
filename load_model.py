# load_model.py
import os
import gdown

MODEL_ID = "1A2B3C4D5EfGHijkL678MNopQR"  # แทนด้วย Google Drive file ID จริง
MODEL_PATH = "model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
