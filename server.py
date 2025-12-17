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
from pydantic import BaseModel

from decoder.ctc_decoder import decode_lattice
from model.wav2vec2 import Wav2Vec2
from phonetics.ipa import symbol_to_descriptor, to_symbol

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


class TranscriptionResponse(BaseModel):
    ipa_text: str
    symbols: List[str]
    descriptors: List[Dict[str, str]]


@app.on_event("startup")
async def load_model():
    global model
    try:
        print("Loading Wav2Vec2 model...")
        model = Wav2Vec2.from_pretrained("pklumpp/Wav2Vec2_CommonPhone")
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")


def convert_audio_to_wav(audio_bytes: bytes, input_format: str = "webm") -> np.ndarray:
    """
    Converts audio bytes to WAV format using ffmpeg.
    """
    with tempfile.NamedTemporaryFile(
        suffix=f".{input_format}", delete=False
    ) as tmp_input:
        tmp_input.write(audio_bytes)
        tmp_input_path = tmp_input.name

    tmp_output_path = tmp_input_path.replace(f".{input_format}", ".wav")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                tmp_input_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                tmp_output_path,
            ],
            check=True,
        )

        try:
            sample_rate, audio_data = wav.read(tmp_output_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

        # Normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128) / 128.0

        return audio_data

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"FFmpeg error: {e}")
    finally:
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)


def run_inference(audio_data: np.ndarray) -> Dict:
    """
    Runs phonetic transcription inference on audio data.
    """
    # Standardize audio
    mean = audio_data.mean()
    std = audio_data.std()
    audio_data = (audio_data - mean) / (std + 1e-9)

    input_tensor = torch.tensor(
        audio_data[np.newaxis, :], dtype=torch.float, device=device
    )

    with torch.no_grad():
        y_pred, enc_features, cnn_features = model(input_tensor)

    phone_sequence, _, _, _ = decode_lattice(
        lattice=y_pred[0].cpu().numpy(),
        enc_feats=enc_features[0].cpu().numpy(),
        cnn_feats=cnn_features[0].cpu().numpy(),
    )

    symbol_sequence = [to_symbol(i) for i in phone_sequence]
    ipa_text = "".join(symbol_sequence)

    descriptors = [
        {"symbol": s, "descriptor": symbol_to_descriptor(s)} for s in symbol_sequence
    ]

    return {
        "ipa_text": ipa_text,
        "symbols": symbol_sequence,
        "descriptors": descriptors,
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts a single audio file and returns IPA phonetic transcription.
    Supports various audio formats (webm, mp3, wav, m4a, etc.)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read the uploaded file
        audio_bytes = await file.read()

        # Determine input format from filename
        file_extension = (
            file.filename.split(".")[-1].lower() if file.filename else "webm"
        )

        # Convert to WAV
        audio_data = convert_audio_to_wav(audio_bytes, file_extension)

        # Run inference
        result = run_inference(audio_data)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "model_loaded": model is not None, "device": device}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
