import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import prediction logic from predict.py
from predict import predict

app = FastAPI(title="ShipsEar API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def api_predict(
    file: UploadFile = File(...),
    model: str = Form("cnn"),
    device: str = Form("cpu")
):
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    # Save uploaded file to a temporary location
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, 'wb') as tmp:
            shutil.copyfileobj(file.file, tmp)

        # Run prediction
        result = predict(tmp_path, model_type=model, device=device)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
