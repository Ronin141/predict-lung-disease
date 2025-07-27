1. รัน server:
   uvicorn app:app --reload

2. ทดสอบ API:
   curl -X POST "http://localhost:8000/predict/" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@path/to/audio/file.wav"

3. หรือใช้ Python requests:
   import requests
   
   with open('audio_file.wav', 'rb') as f:
       response = requests.post(
           'http://localhost:8000/predict/',
           files={'file': f}
       )
   
   result = response.json()
   print(f"Prediction: {result['prediction']}")
   print(f"Confidence: {result['confidence']}")