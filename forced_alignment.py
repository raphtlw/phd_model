import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import nltk
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set up logging
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

# --- Polyfills for WhisperX dependencies to make this standalone ---


def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file to numpy array (mono, 16khz).
    """
    try:
        wav, source_sr = torchaudio.load(file)
        # Mix to mono if necessary
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # Resample if necessary
        if source_sr != sr:
            resampler = torchaudio.transforms.Resample(source_sr, sr)
            wav = resampler(wav)
        return wav.squeeze().numpy()
    except Exception as e:
        raise ValueError(f"Failed to load audio: {e}")


# Simple type definitions to replace whisperx.schema
SingleSegment = Dict[str, Any]  # Expected keys: start, end, text
SingleAlignedSegment = Dict[str, Any]
SingleWordSegment = Dict[str, Any]
SegmentData = Dict[str, Any]

# ------------------------------------------------------------------

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": "nguyenvulebinh/wav2vec2-base-vi-vlsp2020",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
}


def load_align_model(language_code: str, device: str, model_name: Optional[str] = None):
    if model_name is None:
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            # Fallback to English generic if unknown
            model_name = "WAV2VEC2_ASR_BASE_960H"

    # Load NLTK data needed for sentence splitting
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model().to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(f"Could not load model {model_name}: {e}")

        pipeline_type = "huggingface"
        align_model = align_model.to(device)
        align_dictionary = {
            char.lower(): code for char, code in processor.tokenizer.get_vocab().items()
        }

    align_metadata = {
        "language": language_code,
        "dictionary": align_dictionary,
        "type": pipeline_type,
    }
    return align_model, align_metadata


def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
) -> Dict[str, Any]:
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE
    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    total_segments = len(transcript)
    segment_data = {}

    for sdx, segment in enumerate(transcript):
        text = segment["text"]

        # Determine chars to align
        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")

            if char_ in model_dictionary.keys():
                clean_char.append(char_)
                clean_cdx.append(cdx)
            else:
                clean_char.append("*")
                clean_cdx.append(cdx)

        # Sentence tokenizer
        try:
            sentence_splitter = nltk.data.load(f"tokenizers/punkt/{model_lang}.pickle")
        except:
            sentence_splitter = nltk.data.load("tokenizers/punkt/english.pickle")

        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "sentence_spans": sentence_spans,
        }

    aligned_segments = []

    for sdx, segment in enumerate(transcript):
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]

        aligned_seg = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
            "chars": [] if return_char_alignments else None,
        }

        if len(segment_data[sdx]["clean_char"]) == 0:
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment_data[sdx]["clean_char"])
        tokens = [model_dictionary.get(c, -1) for c in text_clean]

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        waveform_segment = audio[:, f1:f2]

        if waveform_segment.shape[-1] < 400:
            lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
            waveform_segment = torch.nn.functional.pad(
                waveform_segment, (0, 400 - waveform_segment.shape[-1])
            )
        else:
            lengths = None

        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(device), lengths=lengths)
            else:
                emissions = model(waveform_segment.to(device)).logits
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()
        blank_id = 0  # Usually 0 in Wav2Vec2

        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack_beam(trellis, emission, tokens, blank_id, beam_width=2)

        if path is None:
            aligned_segments.append(aligned_seg)
            continue

        char_segments = merge_repeats(path, text_clean)
        duration = t2 - t1
        ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

        char_segments_arr = []
        word_idx = 0

        # Align chars
        for cdx, char in enumerate(text):
            start, end, score = None, None, None
            if cdx in segment_data[sdx]["clean_cdx"]:
                clean_idx = segment_data[sdx]["clean_cdx"].index(cdx)
                if clean_idx < len(char_segments):
                    char_seg = char_segments[clean_idx]
                    start = round(char_seg.start * ratio + t1, 3)
                    end = round(char_seg.end * ratio + t1, 3)
                    score = round(char_seg.score, 3)

            char_segments_arr.append(
                {
                    "char": char,
                    "start": start,
                    "end": end,
                    "score": score,
                    "word-idx": word_idx,
                }
            )

            # Increment word index on space or end of string
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx + 1] == " ":
                word_idx += 1

        df_chars = pd.DataFrame(char_segments_arr)

        # Interpolate NaNs for robust timestamps
        df_chars["start"] = df_chars["start"].interpolate(
            method=interpolate_method, limit_direction="both"
        )
        df_chars["end"] = df_chars["end"].interpolate(
            method=interpolate_method, limit_direction="both"
        )

        # Group by word-idx to get word segments
        words = []
        for w_idx, group in df_chars.groupby("word-idx"):
            word_text = "".join(group["char"].tolist()).strip()
            if not word_text:
                continue

            w_start = group["start"].min()
            w_end = group["end"].max()
            w_score = group["score"].mean()

            # Sanity check
            if pd.isna(w_start) or pd.isna(w_end):
                continue

            words.append(
                {
                    "word": word_text,
                    "start": float(w_start),
                    "end": float(w_end),
                    "score": float(w_score) if not pd.isna(w_score) else 0.0,
                }
            )

        aligned_seg["words"] = words
        aligned_segments.append(aligned_seg)

    word_segments = []
    for seg in aligned_segments:
        word_segments.extend(seg["words"])

    return {"segments": aligned_segments, "word_segments": word_segments}


# --- Helpers (Trellis, Backtrack) ---


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + get_wildcard_emission(emission[t], tokens[1:], blank_id),
        )
    return trellis


def get_wildcard_emission(frame_emission, tokens, blank_id):
    assert 0 <= blank_id < len(frame_emission)
    tokens = torch.tensor(tokens) if not isinstance(tokens, torch.Tensor) else tokens
    wildcard_mask = tokens == -1
    regular_scores = frame_emission[tokens.clamp(min=0).long()]
    max_valid_score = frame_emission.clone()
    max_valid_score[blank_id] = float("-inf")
    max_valid_score = max_valid_score.max()
    result = torch.where(wildcard_mask, max_valid_score, regular_scores)
    return result


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class BeamState:
    token_index: int
    time_index: int
    score: float
    path: List[Point]


def backtrack_beam(trellis, emission, tokens, blank_id=0, beam_width=5):
    T, J = trellis.size(0) - 1, trellis.size(1) - 1
    init_state = BeamState(
        token_index=J,
        time_index=T,
        score=trellis[T, J],
        path=[Point(J, T, emission[T, blank_id].exp().item())],
    )
    beams = [init_state]

    while beams and beams[0].token_index > 0:
        next_beams = []
        for beam in beams:
            t, j = beam.time_index, beam.token_index
            if t <= 0:
                continue

            p_stay = emission[t - 1, blank_id]
            p_change = get_wildcard_emission(emission[t - 1], [tokens[j]], blank_id)[0]
            stay_score = trellis[t - 1, j]
            change_score = trellis[t - 1, j - 1] if j > 0 else float("-inf")

            if not math.isinf(stay_score):
                new_path = beam.path.copy()
                new_path.append(Point(j, t - 1, p_stay.exp().item()))
                next_beams.append(
                    BeamState(
                        token_index=j, time_index=t - 1, score=stay_score, path=new_path
                    )
                )

            if j > 0 and not math.isinf(change_score):
                new_path = beam.path.copy()
                new_path.append(Point(j - 1, t - 1, p_change.exp().item()))
                next_beams.append(
                    BeamState(
                        token_index=j - 1,
                        time_index=t - 1,
                        score=change_score,
                        path=new_path,
                    )
                )

        beams = sorted(next_beams, key=lambda x: x.score, reverse=True)[:beam_width]
        if not beams:
            break

    if not beams:
        return None

    best_beam = beams[0]
    t = best_beam.time_index
    j = best_beam.token_index
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        best_beam.path.append(Point(j, t - 1, prob))
        t -= 1

    return best_beam.path[::-1]


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments
