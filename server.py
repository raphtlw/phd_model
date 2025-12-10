import io
import os
import subprocess
import tempfile
from typing import Dict, List

import numpy as np
import scipy.io.wavfile as wav
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scipy import signal

from decoder.ctc_decoder import decode_lattice
from model.wav2vec2 import Wav2Vec2
from phonetics.ipa import symbol_to_descriptor, to_symbol

app = FastAPI(title="Phoneme Recognition API", version="1.0.0")

# CORS middleware to allow Svelte app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Svelte app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device config
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


class TranscriptionResponse(BaseModel):
    ipa_text: str
    symbols: List[str]
    descriptors: List[Dict[str, str]]
    success: bool
    message: str = ""


@app.on_event("startup")
async def load_model():
    """Load model on server startup"""
    global model
    try:
        print("Loading Wav2Vec2 model...")
        model = Wav2Vec2.from_pretrained("pklumpp/Wav2Vec2_CommonPhone")
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


def convert_to_wav(input_bytes: bytes) -> bytes:
    """Convert audio file to WAV format using ffmpeg"""
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_input:
        tmp_input.write(input_bytes)
        tmp_input_path = tmp_input.name

    tmp_output_path = tmp_input_path.replace(".ogg", ".wav")

    try:
        # Use ffmpeg to convert to WAV
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                tmp_input_path,
                "-ar",
                "16000",  # Sample rate
                "-ac",
                "1",  # Mono
                "-y",  # Overwrite output
                tmp_output_path,
            ],
            check=True,
            capture_output=True,
        )

        with open(tmp_output_path, "rb") as f:
            wav_bytes = f.read()

        return wav_bytes
    finally:
        # Cleanup temp files
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)


def process_audio(audio_bytes: bytes, is_wav: bool = True) -> np.ndarray:
    """Process audio bytes into normalized tensor"""
    # Convert to WAV if needed
    if not is_wav:
        audio_bytes = convert_to_wav(audio_bytes)

    # Read wav file from bytes
    audio_io = io.BytesIO(audio_bytes)
    sample_rate, audio_data = wav.read(audio_io)

    # Convert to float
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype == np.uint8:
        audio_data = (audio_data.astype(np.float32) - 128) / 128.0

    # Mix to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16000 if needed
    target_fs = 16000
    if sample_rate != target_fs:
        num_samples = int(len(audio_data) * target_fs / sample_rate)
        audio_data = signal.resample(audio_data, num_samples)

    # Standardize
    mean = audio_data.mean()
    std = audio_data.std()
    audio_data = (audio_data - mean) / (std + 1e-9)

    return audio_data


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "running", "model_loaded": model is not None, "device": device}


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file to IPA phonemes

    Args:
        file: Audio file (WAV, OGG, WebM formats supported)

    Returns:
        JSON with IPA transcription, symbols, and descriptors
    """
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Server may still be starting up."
        )

    try:
        # Read audio file
        audio_bytes = await file.read()

        # Detect if it's a WAV file
        is_wav = audio_bytes[:4] == b"RIFF" or audio_bytes[:4] == b"RF64"

        # Process audio
        audio_data = process_audio(audio_bytes, is_wav=is_wav)

        # Prepare tensor
        input_tensor = torch.tensor(
            audio_data[np.newaxis, :], dtype=torch.float, device=device
        )

        # Run inference
        with torch.no_grad():
            y_pred, enc_features, cnn_features = model(input_tensor)

        # Decode
        phone_sequence, enc_feats, cnn_feats, probs = decode_lattice(
            lattice=y_pred[0].cpu().numpy(),
            enc_feats=enc_features[0].cpu().numpy(),
            cnn_feats=cnn_features[0].cpu().numpy(),
        )

        # Convert to symbols
        symbol_sequence = [to_symbol(i) for i in phone_sequence]
        ipa_text = "".join(symbol_sequence)

        # Get descriptors
        descriptors = [
            {"symbol": s, "descriptor": symbol_to_descriptor(s)}
            for s in symbol_sequence
        ]

        return TranscriptionResponse(
            ipa_text=ipa_text,
            symbols=symbol_sequence,
            descriptors=descriptors,
            success=True,
            message="Transcription completed successfully",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info",
    )
