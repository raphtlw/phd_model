import gc
import json
import logging
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchaudio.transforms as T

# --- Warning Suppression ---
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*Torchaudio's I/O functions now support.*")
warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint.*")
warnings.filterwarnings(
    "ignore", message=".*Some weights of Wav2Vec2 were not initialized.*"
)

from decoder.ctc_decoder import decode_lattice

# Import existing modules
from forced_alignment import align, load_align_model, load_audio
from model.wav2vec2 import Wav2Vec2
from phonetics.ipa import to_symbol

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
SAMPLE_RATE = 16000
MIN_AUDIO_LEN = 320  # Minimum samples for Wav2Vec2
BATCH_SIZE = 1  # Processing one file at a time is safer for stability, though slower


def get_device():
    """Select the best available device (MPS for Mac, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@st.cache_resource
def load_models(device):
    """
    Load both the Alignment model (for word segmentation)
    and the PhD Phoneme Recognition model.
    """
    st.write(f"Loading models on **{device}**...")

    # 1. Load Alignment Model (English)
    align_model, align_metadata = load_align_model("en", str(device))

    # 2. Load Phoneme Recognition Model
    phone_model = Wav2Vec2.from_pretrained("pklumpp/Wav2Vec2_CommonPhone")
    phone_model.to(device)
    phone_model.eval()

    return align_model, align_metadata, phone_model


def process_single_clip(
    row,
    clips_dir: Path,
    align_model,
    align_metadata,
    phone_model,
    device,
    mfcc_transform,
):
    """
    Runs the full pipeline on a single clip:
    1. Load Audio
    2. Forced Alignment (Text -> Audio Word Boundaries)
    3. Segment Extraction & Phoneme Recognition (Audio Segment -> IPA)
    4. MFCC Extraction (Per word segment)
    """
    file_name = row["path"]
    sentence = row["sentence"]
    audio_path = clips_dir / file_name

    if not audio_path.exists():
        return None, "File not found"

    try:
        # --- 1. Load Audio ---
        # forced_alignment.load_audio returns numpy array, we need tensor
        wav_numpy = load_audio(str(audio_path), sr=SAMPLE_RATE)
        wav_tensor = torch.from_numpy(wav_numpy).float().to(device)
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)  # (1, Time)

        # --- 2. Forced Alignment ---
        # We treat the whole sentence as one segment for alignment
        transcript = [
            {"text": sentence, "start": 0.0, "end": wav_tensor.shape[1] / SAMPLE_RATE}
        ]

        alignment_result = align(
            transcript,
            align_model,
            align_metadata,
            wav_tensor,  # Pass tensor directly
            str(device),
            return_char_alignments=False,
        )

        word_segments = alignment_result.get("word_segments", [])

        # --- 3. Word-Level Phoneme & MFCC Pipeline ---
        processed_words = []

        for word_seg in word_segments:
            w_start = word_seg["start"]
            w_end = word_seg["end"]
            w_text = word_seg["word"]

            # Convert time to samples
            start_sample = int(w_start * SAMPLE_RATE)
            end_sample = int(w_end * SAMPLE_RATE)

            # Ensure valid length for model (Wav2Vec2 requires min length)
            if end_sample - start_sample < MIN_AUDIO_LEN:
                processed_words.append(
                    {
                        "word": w_text,
                        "start": w_start,
                        "end": w_end,
                        "phonemes": "",
                        "score": word_seg["score"],
                        "mfcc": None,
                    }
                )
                continue

            # --- Extract Segment ---
            # IMPORTANT: The MFCC is extracted specifically from this split segment below
            segment_wav = wav_tensor[:, start_sample:end_sample]

            # --- Phoneme Recognition ---
            # Standardize Segment (IMPORTANT for the PhD model)
            mean = segment_wav.mean()
            std = segment_wav.std()
            segment_wav_norm = (segment_wav - mean) / (std + 1e-9)

            with torch.no_grad():
                y_pred, _, _ = phone_model(segment_wav_norm)

            # Decode to IPA
            phone_indices, _, _, _ = decode_lattice(y_pred[0].cpu().numpy())
            ipa_symbols = [to_symbol(i) for i in phone_indices]
            ipa_string = " ".join(ipa_symbols)

            # --- MFCC Extraction (Per Word Segment) ---
            # Ensure we use the exact segment extracted above
            mfcc_input = segment_wav
            if mfcc_input.shape[-1] < 400:
                pad_amt = 400 - mfcc_input.shape[-1]
                mfcc_input = torch.nn.functional.pad(mfcc_input, (0, pad_amt))

            mfcc = mfcc_transform(mfcc_input)  # (1, n_mfcc, time)
            mfcc_list = mfcc.squeeze(0).cpu().numpy().tolist()

            processed_words.append(
                {
                    "word": w_text,
                    "start": w_start,
                    "end": w_end,
                    "phonemes": ipa_string,
                    "score": word_seg["score"],
                    "mfcc": mfcc_list,
                }
            )

        return processed_words, None

    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}")
        return None, str(e)


def main():
    st.set_page_config(page_title="CommonVoice Processor", page_icon="🗣️", layout="wide")

    st.title("🗣️ Common Voice 24.0 Word-Phoneme Pipeline")
    st.markdown("""
    This tool processes the **Mozilla Common Voice** dataset to generate:
    1. **Word Alignments** (Time-stamped words)
    2. **Word-Level Phonemes** (IPA transcription per word using `Wav2Vec2_CommonPhone`)
    3. **Word-Level MFCC Features**

    It outputs separate CSV files for each accent found in the dataset.
    **Note:** The output format is one row per word.
    """)

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")

    # Device Selection
    device = get_device()
    st.sidebar.info(f"Detected Device: **{device}**")

    # Path Inputs
    default_path = os.getcwd()
    dataset_path_str = st.sidebar.text_input("Dataset Root Folder", value=default_path)
    output_path_str = st.sidebar.text_input(
        "Output CSV Folder", value=os.path.join(default_path, "processed_data")
    )

    st.sidebar.divider()
    st.sidebar.subheader("Subset Selection")
    st.sidebar.markdown("Settings apply to **EACH** selected accent individually.")

    limit_data = st.sidebar.checkbox("Limit Number of Speakers", value=True)
    n_speakers_per_accent = 10
    if limit_data:
        n_speakers_per_accent = st.sidebar.number_input(
            "Max Speakers per Accent (Random)", min_value=1, value=10, step=1
        )

    st.sidebar.write("---")

    limit_clips = st.sidebar.checkbox(
        "Limit Clips per Speaker",
        value=True,
        help="If unchecked, ALL recordings for the selected speakers will be processed.",
    )
    n_clips_per_speaker = 5
    if limit_clips:
        n_clips_per_speaker = st.sidebar.number_input(
            "Max Clips per Speaker (Random)", min_value=1, value=5, step=1
        )

    dataset_path = Path(dataset_path_str)
    clips_path = dataset_path / "clips"
    validated_tsv_path = dataset_path / "validated.tsv"

    # --- Main UI ---

    # 1. Validation & Loading
    if not dataset_path.exists():
        st.error(f"Dataset path does not exist: {dataset_path}")
        return

    if not validated_tsv_path.exists():
        st.error(f"Could not find `validated.tsv` in {dataset_path}")
        return

    if "df" not in st.session_state:
        st.session_state.df = None

    if st.button("Load Dataset Metadata"):
        with st.spinner("Loading validated.tsv..."):
            try:
                # FIX: Explicitly set dtypes for columns that might be mixed
                df = pd.read_csv(
                    validated_tsv_path,
                    sep="\t",
                    dtype={
                        "client_id": str,
                        "path": str,
                        "sentence": str,
                        "accents": str,
                        "sentence_domain": str,
                        "gender": str,
                        "age": str,
                        "locale": str,
                    },
                    low_memory=False,
                )
                st.session_state.df = df
                st.success(f"Loaded {len(df):,} clips.")
            except Exception as e:
                st.error(f"Error reading TSV: {e}")

    # 2. Analysis & Processing
    if st.session_state.df is not None:
        df = st.session_state.df

        # Analyze Accents
        st.subheader("Dataset Analysis")

        # Fill NaN accents with 'Unknown'
        df["accents"] = df["accents"].fillna("Unknown")

        accent_counts = df["accents"].value_counts()

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(accent_counts, height=300)
        with c2:
            st.bar_chart(accent_counts.head(20))  # Show top 20

        st.divider()
        st.subheader("Run Pipeline")

        selected_accents = st.multiselect(
            "Select Accents to Process (Leave empty for ALL)",
            options=accent_counts.index.tolist(),
        )

        if st.button("🚀 Start Processing", type="primary"):
            # --- 1. Identify Processed Files (Resume Logic) ---
            processed_paths = set()
            out_dir_path = Path(output_path_str)
            if out_dir_path.exists():
                with st.spinner("Scanning for existing processed files to resume..."):
                    for csv_file in out_dir_path.glob("common_voice_*.csv"):
                        try:
                            # We only need the 'path' column to identify what's done
                            existing_df = pd.read_csv(csv_file, usecols=["path"])
                            processed_paths.update(existing_df["path"].tolist())
                        except Exception:
                            # File might be empty or corrupt, ignore
                            pass

            if processed_paths:
                st.info(
                    f"♻️ Resuming: Found {len(processed_paths)} clips already processed. They will be skipped."
                )

            # --- 2. Build Target DataFrame (Per Accent Logic) ---
            final_dfs = []

            # Use all accents if none selected
            target_accents = (
                selected_accents if selected_accents else accent_counts.index.tolist()
            )

            with st.spinner("Building randomized subset..."):
                for accent in target_accents:
                    # Filter for accent
                    accent_df = df[df["accents"] == accent]

                    # 1. Filter out already processed files
                    accent_df = accent_df[~accent_df["path"].isin(processed_paths)]

                    if accent_df.empty:
                        continue

                    # 2. Randomly select speakers
                    if limit_data:
                        unique_speakers = accent_df["client_id"].unique()
                        if len(unique_speakers) > n_speakers_per_accent:
                            selected_spks = np.random.choice(
                                unique_speakers,
                                size=n_speakers_per_accent,
                                replace=False,
                            )
                            accent_df = accent_df[
                                accent_df["client_id"].isin(selected_spks)
                            ]

                    # 3. Randomly select clips per speaker
                    if limit_clips:
                        accent_df = (
                            accent_df.groupby("client_id")
                            .apply(
                                lambda x: x.sample(n=min(len(x), n_clips_per_speaker))
                            )
                            .reset_index(drop=True)
                        )

                    final_dfs.append(accent_df)

            if not final_dfs:
                st.warning(
                    "No clips found to process! Either the selection is empty or all clips have already been processed."
                )
                return

            target_df = pd.concat(final_dfs)
            st.write(
                f"Processing **{len(target_df):,}** clips across {target_df['accents'].nunique()} accents..."
            )

            # Create output dir
            os.makedirs(output_path_str, exist_ok=True)

            # Load Models
            try:
                align_model, align_meta, phone_model = load_models(device)

                # Init MFCC
                mfcc_transform = T.MFCC(
                    sample_rate=SAMPLE_RATE,
                    n_mfcc=13,
                    melkwargs={
                        "n_fft": 400,
                        "hop_length": 160,
                        "n_mels": 23,
                        "center": False,
                    },
                ).to(device)
            except Exception as e:
                st.error(f"Failed to load models: {e}")
                return

            # Group by accent to write separate files
            groups = target_df.groupby("accents")

            total_progress = st.progress(0)
            overall_counter = 0
            total_clips = len(target_df)

            status_container = st.empty()
            file_log_container = st.empty()  # Container for rapid file logging

            for accent, group_df in groups:
                # Sanitize accent filename
                safe_accent = "".join([c if c.isalnum() else "_" for c in accent])
                output_csv = Path(output_path_str) / f"common_voice_{safe_accent}.csv"

                status_container.markdown(
                    f"### Processing Accent: `{accent}` ({len(group_df)} clips)"
                )

                # Prepare batch list
                batch_results = []
                BATCH_WRITE_FREQ = 20  # Lower frequency for safety

                # Iterate rows
                for idx, row in group_df.iterrows():
                    # --- Log current file to UI ---
                    file_log_container.code(f"Processing: {row['path']}")

                    words_data, error = process_single_clip(
                        row,
                        clips_path,
                        align_model,
                        align_meta,
                        phone_model,
                        device,
                        mfcc_transform,
                    )

                    if words_data:
                        # Iterate through each processed word and create a row
                        for w_data in words_data:
                            out_row = row.to_dict()

                            # Add flattened word data
                            out_row["word"] = w_data["word"]
                            out_row["start"] = w_data["start"]
                            out_row["end"] = w_data["end"]
                            out_row["phonemes"] = w_data["phonemes"]
                            out_row["word_score"] = w_data["score"]

                            # Add MFCC for this specific word
                            out_row["mfcc_json"] = json.dumps(w_data["mfcc"])

                            out_row["processing_error"] = None
                            batch_results.append(out_row)
                    else:
                        # Log error row so we know this file failed
                        out_row = row.to_dict()
                        out_row["word"] = None
                        out_row["start"] = None
                        out_row["end"] = None
                        out_row["phonemes"] = None
                        out_row["word_score"] = None
                        out_row["mfcc_json"] = None
                        out_row["processing_error"] = error
                        batch_results.append(out_row)

                    overall_counter += 1

                    # Update Progress
                    if overall_counter % 5 == 0:
                        total_progress.progress(min(overall_counter / total_clips, 1.0))

                    # Batch Write
                    if len(batch_results) >= BATCH_WRITE_FREQ:
                        batch_df = pd.DataFrame(batch_results)

                        # Reorder columns: word and accent side-by-side
                        cols = batch_df.columns.tolist()
                        priority_cols = ["path", "word", "accents", "phonemes"]
                        # Filter to ensure they exist
                        priority_cols = [c for c in priority_cols if c in cols]
                        remaining_cols = [c for c in cols if c not in priority_cols]

                        batch_df = batch_df[priority_cols + remaining_cols]

                        # Append to CSV
                        write_header = not output_csv.exists()
                        batch_df.to_csv(
                            output_csv, mode="a", header=write_header, index=False
                        )
                        batch_results = []  # Clear memory

                    # --- Memory Cleanup ---
                    # Periodically clear GPU cache and Garbage collect
                    if overall_counter % 20 == 0:
                        clear_memory()

                # Write remaining items for this accent
                if batch_results:
                    batch_df = pd.DataFrame(batch_results)

                    cols = batch_df.columns.tolist()
                    priority_cols = ["path", "word", "accents", "phonemes"]
                    priority_cols = [c for c in priority_cols if c in cols]
                    remaining_cols = [c for c in cols if c not in priority_cols]
                    batch_df = batch_df[priority_cols + remaining_cols]

                    write_header = not output_csv.exists()
                    batch_df.to_csv(
                        output_csv, mode="a", header=write_header, index=False
                    )

                st.toast(f"Finished accent: {accent} -> {output_csv.name}")
                # Clear memory after accent block
                clear_memory()

            total_progress.progress(1.0)
            file_log_container.success("Done!")
            st.success(f"Processing Complete! Files saved to `{output_path_str}`")


if __name__ == "__main__":
    main()
