#!/usr/bin/env python3
"""Wav2Vec2_CommonPhone phoneme transcription server.

This FastAPI server accepts browser-recorded audio (e.g. `audio/webm;codecs=opus`),
converts it to 16kHz mono WAV via ffmpeg, and runs phonetic transcription using:

- pklumpp/Wav2Vec2_CommonPhone

Response format is kept compatible with the existing Svelte frontend:
- ipa_text: string (space-separated IPA symbols for display)
- symbols: list[str] (IPA symbols for downstream accent analysis)
"""

import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from decoder.ctc_decoder import decode_lattice
from model.wav2vec2 import Wav2Vec2
from phonetics.ipa import strip_pauses, symbol_to_descriptor, to_symbol


def get_best_device() -> str:
    """Get the best available device: CUDA > MPS > CPU."""

    if torch.cuda.is_available():
        return "cuda"

    # Apple Silicon
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


DEVICE = get_best_device()
ml_models: Dict[str, Any] = {}


class TranscriptionSegment(BaseModel):
    start_ms: int
    end_ms: int
    ipa_text: str
    symbols: List[str]
    descriptors: List[Dict[str, str]]


class TranscriptionWord(BaseModel):
    word: str
    start_ms: int
    end_ms: int
    ipa_text: str
    symbols: List[str]
    descriptors: List[Dict[str, str]]


class TranscriptionResponse(BaseModel):
    ipa_text: str
    symbols: List[str]
    descriptors: List[Dict[str, str]]

    # Optional pause-based segmentation (best-effort). Note: these are *audio chunks*,
    # not guaranteed true lexical word boundaries.
    segments: Optional[List[TranscriptionSegment]] = None

    # Optional fixed-prompt word alignment (best-effort). Each prompt word gets one
    # audio chunk by splitting/merging pause segments.
    words: Optional[List[TranscriptionWord]] = None

    # Convenience display formats
    segmented_ipa_text: Optional[str] = None
    word_ipa_text: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print(f"Loading Wav2Vec2_CommonPhone model on {DEVICE}...")
        model = Wav2Vec2.from_pretrained("pklumpp/Wav2Vec2_CommonPhone")
        model.to(DEVICE)
        model.eval()
        ml_models["wav2vec2"] = model
        print("Model loaded successfully!")
    except Exception as exc:
        print(f"Error loading model: {exc}")
        raise

    yield

    ml_models.clear()
    print("Model unloaded and resources released")


app = FastAPI(title="Wav2Vec2_CommonPhone Transcription API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def apply_voice_focus_preprocessing(
    audio_data: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Apply light voice-focused cleanup.

    This is intentionally lightweight (no extra deps):
    - Bandpass filter to typical voice range
    - Simple spectral noise gate / subtraction

    Knobs are controlled via environment variables:
    - `VOICE_NR_ENABLED` (default: "1")
    - `VOICE_NR_BANDPASS_LOW_HZ` (default: "80")
    - `VOICE_NR_BANDPASS_HIGH_HZ` (default: "7800")
    - `VOICE_NR_STRENGTH` (default: "0.8", 0..1)
    - `VOICE_NR_NOISE_SEC` (default: "0.25")
    """

    enabled = os.getenv("VOICE_NR_ENABLED", "1").lower() not in {"0", "false", "no"}
    if not enabled:
        return audio_data

    try:
        low_hz = float(os.getenv("VOICE_NR_BANDPASS_LOW_HZ", "80"))
        high_hz = float(os.getenv("VOICE_NR_BANDPASS_HIGH_HZ", "7800"))
        strength = float(os.getenv("VOICE_NR_STRENGTH", "0.8"))
        noise_sec = float(os.getenv("VOICE_NR_NOISE_SEC", "0.25"))
    except ValueError:
        # If env vars are invalid, fall back safely.
        low_hz, high_hz, strength, noise_sec = 80.0, 7800.0, 0.8, 0.25

    strength = float(np.clip(strength, 0.0, 1.0))

    if audio_data.size < int(0.05 * sample_rate):
        return audio_data

    # 1) Bandpass (voice focus)
    nyquist = sample_rate / 2.0
    low = max(1.0, min(low_hz, nyquist - 1.0)) / nyquist
    high = max(low + 1e-6, min(high_hz, nyquist - 1.0)) / nyquist
    try:
        sos = signal.butter(6, [low, high], btype="bandpass", output="sos")
        audio_data = signal.sosfiltfilt(sos, audio_data).astype(np.float32, copy=False)
    except Exception:
        # Filtering is a best-effort enhancement.
        pass

    # 2) Spectral subtraction / noise gate
    # Estimate noise from the first `noise_sec` seconds (common for browser recordings).
    # Use Hann + 50% overlap to satisfy NOLA/COLA for istft.
    nperseg = max(128, int(0.02 * sample_rate))  # ~20ms
    noverlap = nperseg // 2

    try:
        # Explicitly use a DFT-even Hann window (aka "periodic" Hann).
        # This avoids SciPy's NOLA warning during `istft`.
        stft_window = signal.get_window("hann", nperseg, fftbins=True)

        _, _, stft_matrix = signal.stft(
            audio_data,
            fs=sample_rate,
            window=stft_window,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary="zeros",
            padded=True,
        )

        magnitude = np.abs(stft_matrix)
        phase = np.exp(1j * np.angle(stft_matrix))

        noise_frames = max(1, int((noise_sec * sample_rate - nperseg) / (nperseg - noverlap) + 1))
        noise_profile = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)

        # Subtract a scaled noise profile; keep a small floor.
        cleaned_mag = np.maximum(magnitude - strength * noise_profile, 0.0)
        gain_floor = 0.05
        gain = np.clip(cleaned_mag / (magnitude + 1e-10), gain_floor, 1.0)

        cleaned_stft = magnitude * gain * phase
        _, cleaned = signal.istft(
            cleaned_stft,
            fs=sample_rate,
            window=stft_window,
            nperseg=nperseg,
            noverlap=noverlap,
            input_onesided=True,
            boundary=True,
        )

        cleaned = np.asarray(cleaned, dtype=np.float32).reshape(-1)
        if cleaned.size:
            # Match original length.
            cleaned = cleaned[: audio_data.shape[0]]
            audio_data = cleaned
    except Exception:
        pass

    # Safety clip (avoid huge values after filtering)
    audio_data = np.clip(audio_data, -1.0, 1.0).astype(np.float32, copy=False)
    return audio_data


def convert_audio_to_wav(audio_bytes: bytes, input_format: str = "webm") -> np.ndarray:
    """Convert audio bytes to 16kHz mono float32 waveform."""

    with tempfile.NamedTemporaryFile(
        suffix=f".{input_format}", delete=False
    ) as tmp_input:
        tmp_input.write(audio_bytes)
        tmp_input_path = tmp_input.name

    tmp_output_path = tmp_input_path.rsplit(".", 1)[0] + ".wav"

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

        sample_rate, audio_data = wav.read(tmp_output_path)
        if sample_rate != 16000:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 16kHz after conversion, got {sample_rate}Hz",
            )

        # Normalize integer PCM to [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
        else:
            audio_data = audio_data.astype(np.float32)

        # Ensure shape is (samples,)
        audio_data = np.asarray(audio_data).reshape(-1)

        audio_data = apply_voice_focus_preprocessing(audio_data, sample_rate)
        return audio_data

    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=400, detail=f"FFmpeg error: {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {exc}")
    finally:
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)


def _parse_prompt_words(prompt_text: str) -> List[str]:
    # Keep it simple and dependency-free.
    words: List[str] = []
    for raw in prompt_text.strip().split():
        w = "".join(ch for ch in raw.lower() if ch.isalnum() or ch in {"'"})
        if w:
            words.append(w)
    return words


def segment_audio_to_n_chunks_by_energy(
    audio_data: np.ndarray,
    sample_rate: int,
    n_chunks: int,
) -> List[Tuple[int, int]]:
    """Split audio into `n_chunks` contiguous chunks using energy minima.

    This is designed for fixed-prompt "word-level" mode where speakers may not
    pause clearly between words.

    Env knobs:
    - `VOICE_WORD_MIN_MS` (default: "120")
    - `VOICE_WORD_SMOOTH_MS` (default: "60")
    - `VOICE_WORD_EDGE_MS` (default: "80")
    """

    if n_chunks <= 0:
        return []

    total = int(audio_data.size)
    if total <= 0:
        return []

    if n_chunks == 1:
        return [(0, total)]

    try:
        min_ms = int(os.getenv("VOICE_WORD_MIN_MS", "120"))
        smooth_ms = int(os.getenv("VOICE_WORD_SMOOTH_MS", "60"))
        edge_ms = int(os.getenv("VOICE_WORD_EDGE_MS", "80"))
    except ValueError:
        min_ms, smooth_ms, edge_ms = 120, 60, 80

    min_len = max(1, int(min_ms * sample_rate / 1000))
    edge = max(0, int(edge_ms * sample_rate / 1000))

    # Frame RMS (20ms window, 10ms hop)
    frame_len = max(128, int(0.02 * sample_rate))
    hop_len = max(64, int(0.01 * sample_rate))

    if total < frame_len + hop_len:
        # Too short: equal split
        step = max(1, total // n_chunks)
        ranges = [(i * step, total if i == n_chunks - 1 else (i + 1) * step) for i in range(n_chunks)]
        return ranges

    num_frames = 1 + (total - frame_len) // hop_len
    frame_starts = (np.arange(num_frames) * hop_len).astype(np.int64)
    sample_offsets = np.arange(frame_len, dtype=np.int64)
    frames = audio_data[frame_starts[:, None] + sample_offsets[None, :]]

    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)

    # Smooth energy a bit to reduce spurious minima.
    smooth_len = max(1, int(smooth_ms / 10))  # hop is ~10ms
    if smooth_len > 1:
        kernel = np.ones(smooth_len, dtype=np.float32) / float(smooth_len)
        rms = np.convolve(rms.astype(np.float32), kernel, mode="same")

    # Candidate boundary frame indices: low energy points.
    # We select N-1 boundaries with spacing constraints.
    boundary_count = n_chunks - 1

    # Convert min_len in samples to frames.
    min_gap_frames = max(1, int(min_len / hop_len))
    edge_frames = int(edge / hop_len)

    valid_start = edge_frames
    valid_end = max(valid_start + 1, len(rms) - edge_frames)

    if valid_end - valid_start <= boundary_count:
        # Not enough room; fall back to equal split.
        step = max(1, total // n_chunks)
        return [(i * step, total if i == n_chunks - 1 else (i + 1) * step) for i in range(n_chunks)]

    candidate_idx = np.arange(valid_start, valid_end)
    candidate_energy = rms[valid_start:valid_end]
    order = np.argsort(candidate_energy)  # lowest first

    selected: List[int] = []
    for rel_i in order.tolist():
        idx = int(candidate_idx[rel_i])
        if any(abs(idx - s) < min_gap_frames for s in selected):
            continue
        selected.append(idx)
        if len(selected) >= boundary_count:
            break

    if len(selected) < boundary_count:
        # Fall back to approximately-equal boundaries (in frame space)
        selected = [int(valid_start + (i + 1) * (valid_end - valid_start) / n_chunks) for i in range(boundary_count)]

    selected = sorted(selected)

    # Convert frame indices to sample boundaries.
    boundaries = [0] + [int(i * hop_len) for i in selected] + [total]

    # Enforce monotonicity and minimum length by nudging boundaries.
    ranges: List[Tuple[int, int]] = []
    prev = boundaries[0]
    for b in boundaries[1:]:
        b = max(b, prev + 1)
        ranges.append((prev, b))
        prev = b

    # Ensure we got exactly n_chunks
    if len(ranges) != n_chunks:
        step = max(1, total // n_chunks)
        return [(i * step, total if i == n_chunks - 1 else (i + 1) * step) for i in range(n_chunks)]

    # Apply min_len by merging forward if needed.
    fixed: List[Tuple[int, int]] = []
    i = 0
    while i < len(ranges):
        s, e = ranges[i]
        while (e - s) < min_len and i + 1 < len(ranges):
            _, e2 = ranges[i + 1]
            e = e2
            i += 1
        fixed.append((s, e))
        i += 1

    # If merging reduced chunk count, re-split equally to requested n_chunks.
    if len(fixed) != n_chunks:
        step = max(1, total // n_chunks)
        return [(i * step, total if i == n_chunks - 1 else (i + 1) * step) for i in range(n_chunks)]

    return fixed


def segment_audio_by_silence(
    audio_data: np.ndarray,
    sample_rate: int,
) -> List[Tuple[int, int]]:
    """Split audio into speech chunks based on silences.

    This is a best-effort segmentation to make downstream phoneme-distance
    computations easier. These segments are *not guaranteed* to align to real
    lexical word boundaries (for that you'd need an ASR model or forced alignment
    against a known transcript).

    Env knobs:
    - `VOICE_SEG_SILENCE_MS` (default: "250")
    - `VOICE_SEG_MIN_MS` (default: "120")
    - `VOICE_SEG_PAD_MS` (default: "40")
    - `VOICE_SEG_RMS_MULT` (default: "3.0")
    - `VOICE_SEG_RMS_PCTL` (default: "20")
    """

    try:
        silence_ms = int(os.getenv("VOICE_SEG_SILENCE_MS", "250"))
        min_ms = int(os.getenv("VOICE_SEG_MIN_MS", "120"))
        pad_ms = int(os.getenv("VOICE_SEG_PAD_MS", "40"))
        rms_mult = float(os.getenv("VOICE_SEG_RMS_MULT", "3.0"))
        rms_pctl = float(os.getenv("VOICE_SEG_RMS_PCTL", "20"))
    except ValueError:
        silence_ms, min_ms, pad_ms, rms_mult, rms_pctl = 250, 120, 40, 3.0, 20.0

    frame_len = max(128, int(0.02 * sample_rate))  # 20ms
    hop_len = max(64, int(0.01 * sample_rate))  # 10ms

    if audio_data.size <= frame_len:
        return [(0, int(audio_data.size))]

    num_frames = 1 + max(0, (audio_data.size - frame_len) // hop_len)
    frame_starts = (np.arange(num_frames) * hop_len).astype(np.int64)
    sample_offsets = np.arange(frame_len, dtype=np.int64)

    frames = audio_data[frame_starts[:, None] + sample_offsets[None, :]]
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)

    noise_floor = float(np.percentile(rms, np.clip(rms_pctl, 0.0, 100.0)))
    threshold = max(noise_floor * max(rms_mult, 1.0), 1e-4)

    speech = rms > threshold

    # Find contiguous speech runs in frame space.
    segments: List[Tuple[int, int]] = []
    in_run = False
    run_start = 0
    for i, is_speech in enumerate(speech.tolist()):
        if is_speech and not in_run:
            in_run = True
            run_start = i
        elif not is_speech and in_run:
            in_run = False
            segments.append((run_start, i - 1))
    if in_run:
        segments.append((run_start, len(speech) - 1))

    if not segments:
        return [(0, int(audio_data.size))]

    pad = int(pad_ms * sample_rate / 1000)
    min_len = int(min_ms * sample_rate / 1000)
    merge_gap = int(silence_ms * sample_rate / 1000)

    # Convert frame runs to sample ranges and pad a bit.
    sample_segments: List[Tuple[int, int]] = []
    for start_f, end_f in segments:
        start_s = max(0, int(start_f * hop_len) - pad)
        end_s = min(int(audio_data.size), int(end_f * hop_len + frame_len) + pad)
        if end_s - start_s >= min_len:
            sample_segments.append((start_s, end_s))

    if not sample_segments:
        return [(0, int(audio_data.size))]

    # Merge close-by segments.
    merged: List[Tuple[int, int]] = [sample_segments[0]]
    for start_s, end_s in sample_segments[1:]:
        prev_start, prev_end = merged[-1]
        if start_s - prev_end <= merge_gap:
            merged[-1] = (prev_start, max(prev_end, end_s))
        else:
            merged.append((start_s, end_s))

    return merged


def run_inference(audio_data: np.ndarray, *, remove_pauses: bool = False) -> Dict:
    model = ml_models.get("wav2vec2")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if audio_data.size < 10:
        raise HTTPException(status_code=400, detail="Audio too short")

    # IMPORTANT: Always standardize input audio
    mean = float(audio_data.mean())
    std = float(audio_data.std())
    audio_data = (audio_data - mean) / (std + 1e-9)

    input_tensor = torch.tensor(
        audio_data[np.newaxis, :], dtype=torch.float, device=DEVICE
    )

    with torch.no_grad():
        y_pred, enc_features, cnn_features = model(input_tensor)

    phone_sequence, _, _, _ = decode_lattice(
        lattice=y_pred[0].cpu().numpy(),
        enc_feats=enc_features[0].cpu().numpy(),
        cnn_feats=cnn_features[0].cpu().numpy(),
    )

    symbol_sequence = [to_symbol(i) for i in phone_sequence]
    if remove_pauses:
        symbol_sequence = strip_pauses(symbol_sequence)

    return {
        "ipa_text": " ".join(symbol_sequence),
        "symbols": symbol_sequence,
        "descriptors": [
            {"symbol": s, "descriptor": symbol_to_descriptor(s)}
            for s in symbol_sequence
        ],
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    segmented: bool = False,
    word_level: bool = False,
    remove_pauses: bool = Query(
        default=False,
        description="If true, removes pause tokens like '(...)' from IPA outputs.",
    ),
    prompt: Optional[str] = Query(
        default=None,
        description=(
            "Fixed prompt text. If omitted, uses env FIXED_PROMPT_TEXT. "
            "Used only when word_level=true."
        ),
    ),
):
    """Transcribe audio to IPA.

    - `segmented=true`: returns pause-delimited chunks.
    - `word_level=true`: assumes a fixed prompt and returns one IPA chunk per prompt
      word by splitting/merging pause segments (best-effort).
    - `remove_pauses=true`: removes pause tokens like `(...)` from IPA output.
    """

    try:
        audio_bytes = await file.read()
        file_extension = (
            file.filename.split(".")[-1].lower() if file.filename else "webm"
        )
        audio_data = convert_audio_to_wav(audio_bytes, file_extension)

        sample_rate = 16000

        # Default behavior (backwards compatible)
        if not segmented and not word_level:
            return run_inference(audio_data, remove_pauses=remove_pauses)

        ranges = segment_audio_by_silence(audio_data, sample_rate)

        segments: List[Dict[str, Any]] = []
        segment_display_parts: List[str] = []

        if segmented:
            for start_s, end_s in ranges:
                segment_audio = audio_data[start_s:end_s]
                if segment_audio.size < 10:
                    continue

                segment_result = run_inference(segment_audio, remove_pauses=remove_pauses)
                segments.append(
                    {
                        "start_ms": int(start_s * 1000 / sample_rate),
                        "end_ms": int(end_s * 1000 / sample_rate),
                        **segment_result,
                    }
                )
                if remove_pauses:
                    segment_display_parts.append(f"/{segment_result['ipa_text']}/")
                else:
                    segment_display_parts.append(f"/(...) {segment_result['ipa_text']} (...)/")

        words_payload: Optional[List[Dict[str, Any]]] = None
        word_display_parts: List[str] = []

        if word_level:
            prompt_text = prompt or os.getenv("FIXED_PROMPT_TEXT", "").strip()
            if not prompt_text:
                raise HTTPException(
                    status_code=400,
                    detail="word_level=true requires `prompt` or env FIXED_PROMPT_TEXT",
                )

            prompt_words = _parse_prompt_words(prompt_text)
            if not prompt_words:
                raise HTTPException(status_code=400, detail="Prompt has no words")

            # Prefer silence-based ranges when they match word count.
            # Otherwise fall back to energy-minima splitting to get one chunk per word.
            if len(ranges) == len(prompt_words):
                word_ranges = ranges
            else:
                word_ranges = segment_audio_to_n_chunks_by_energy(
                    audio_data, sample_rate, len(prompt_words)
                )

            words_payload = []
            for (start_s, end_s), word in zip(word_ranges, prompt_words):
                segment_audio = audio_data[start_s:end_s]
                if segment_audio.size < 10:
                    segment_result = {"ipa_text": "", "symbols": [], "descriptors": []}
                else:
                    segment_result = run_inference(
                        segment_audio, remove_pauses=remove_pauses
                    )

                words_payload.append(
                    {
                        "word": word,
                        "start_ms": int(start_s * 1000 / sample_rate),
                        "end_ms": int(end_s * 1000 / sample_rate),
                        **segment_result,
                    }
                )
                word_display_parts.append(word)
                if remove_pauses:
                    word_display_parts.append(f"/{segment_result['ipa_text']}/")
                else:
                    word_display_parts.append(f"/(...) {segment_result['ipa_text']} (...)/")

        full_result = run_inference(audio_data, remove_pauses=remove_pauses)
        full_result["segments"] = segments or None
        full_result["segmented_ipa_text"] = (
            "\n".join(segment_display_parts) if segment_display_parts else None
        )
        full_result["words"] = words_payload
        full_result["word_ipa_text"] = "\n".join(word_display_parts) if word_display_parts else None
        return full_result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": "wav2vec2" in ml_models,
        "device": DEVICE,
    }


if __name__ == "__main__":
    # NOTE: Uvicorn requires an import string for `reload`/`workers`.
    # This avoids the warning: "You must pass the application as an import string..."
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
