"""
Streamlit App: Word Alignment & Phoneme Recognition Pipeline (Batch Processing)
Processes large audio archives in batches to minimize disk/memory usage.

Usage:
    streamlit run word_alignment_app.py
"""

import gc
import io
import json
import logging
import os
import random
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
import torchaudio
import torchaudio.transforms as T

from decoder.ctc_decoder import decode_lattice

# --- Import Project Modules ---
from forced_alignment import align, load_align_model, load_audio
from model.wav2vec2 import Wav2Vec2
from phonetics.ipa import to_symbol

# --- Configuration ---
SAMPLE_RATE = 16000
MIN_AUDIO_LEN = 320  # Minimum samples for Wav2Vec2
BATCH_SIZE_FILES = 50  # If no nested zips, process this many wavs at once

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device():
    """Select the best available device."""
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
def load_models(device_str: str):
    """
    Load Alignment Model and PhD Phoneme Recognition Model.
    Cached to avoid reloading on every run.
    """
    device = torch.device(device_str)
    st.write(f"Loading models on **{device}**...")

    # 1. Load Alignment Model
    align_model, align_metadata = load_align_model("en", device_str)

    # 2. Load Phoneme Recognition Model
    try:
        phone_model = Wav2Vec2.from_pretrained("pklumpp/Wav2Vec2_CommonPhone")
        phone_model.to(device)
        phone_model.eval()
    except Exception as e:
        st.error(f"Error loading phoneme model: {e}")
        return None, None, None

    return align_model, align_metadata, phone_model


class ZipBatchProcessor:
    """
    Handles processing of a large zip file in batches.
    - Scans structure without extracting.
    - Extracts metadata (TXT) globally.
    - Yields temporary directories containing chunks of data.
    - Cleans up temporary chunks immediately.
    """

    def __init__(self, source: Union[str, Path, io.BytesIO], temp_root: Path):
        self.source = source
        self.temp_root = temp_root
        self.zip_ref = None
        self.is_path = isinstance(source, (str, Path))

        # Open the zip file object
        if self.is_path:
            self.zip_ref = zipfile.ZipFile(source, "r")
        else:
            self.zip_ref = zipfile.ZipFile(source, "r")

        self.all_files = self.zip_ref.namelist()

    def close(self):
        if self.zip_ref:
            self.zip_ref.close()

    def filter_dataset(
        self, n_speakers: int, n_clips_per_speaker: int, limit_clips: bool = True
    ):
        """
        Filters self.all_files to include a random subset of speakers
        and a random subset of clips per speaker.
        """
        # Separate WAVs/ZIPs from other files (TXTs, metadata)
        # We want to keep metadata, but filter the audio sources
        audio_files = []
        other_files = []

        for f in self.all_files:
            lower_f = f.lower()
            if lower_f.endswith(".wav") or lower_f.endswith(".zip"):
                if not f.startswith("._") and "__MACOSX" not in f:
                    audio_files.append(f)
            else:
                other_files.append(f)

        # Group by Speaker ID (heuristic: first 6 chars)
        speaker_map = defaultdict(list)
        for f in audio_files:
            # Simple heuristic: filename usually starts with speaker ID
            # e.g., 000010_script.wav -> 000010
            # If path has directories, take the filename part
            fname = Path(f).name
            if len(fname) >= 6:
                spk_id = fname[:6]
                speaker_map[spk_id].append(f)
            else:
                speaker_map["unknown"].append(f)

        all_speakers = list(speaker_map.keys())

        # 1. Randomly select speakers
        if n_speakers < len(all_speakers):
            selected_speakers = random.sample(all_speakers, n_speakers)
        else:
            selected_speakers = all_speakers

        # 2. Randomly select clips for each selected speaker
        final_audio_files = []

        for spk in selected_speakers:
            clips = speaker_map[spk]
            if limit_clips and n_clips_per_speaker < len(clips):
                selected_clips = random.sample(clips, n_clips_per_speaker)
                final_audio_files.extend(selected_clips)
            else:
                # Select ALL clips if limit_clips is False
                final_audio_files.extend(clips)

        # Reconstruct all_files
        # We keep ALL other files (scripts/metadata) to ensure we don't break transcript matching
        self.all_files = other_files + final_audio_files

        return len(selected_speakers), len(final_audio_files)

    def extract_scripts(self) -> Dict[str, str]:
        """
        Extracts all .txt files to a persistent metadata folder and parses them.
        Returns the scripts dictionary.
        """
        script_dir = self.temp_root / "metadata_scripts"
        script_dir.mkdir(exist_ok=True)

        txt_files = [
            f
            for f in self.all_files
            if f.lower().endswith(".txt") and not f.startswith("._")
        ]

        scripts = {}
        if not txt_files:
            return scripts

        # Extract all text files
        for f in txt_files:
            self.zip_ref.extract(f, path=script_dir)

        # Parse them
        scripts = parse_script_files(script_dir)
        return scripts

    def get_batches(self) -> List[dict]:
        """
        Determines the batch strategy.
        1. If nested ZIPs found (e.g. WAVE/Speaker.zip), each zip is a batch.
        2. Else, groups WAV files into chunks.
        """
        # Filter all_files based on current state (filtered or not)
        nested_zips = [
            f
            for f in self.all_files
            if f.lower().endswith(".zip") and not f.startswith("._")
        ]

        batches = []

        if nested_zips:
            # Strategy 1: Nested Zips
            for z in nested_zips:
                batches.append({"type": "nested_zip", "target": z, "id": Path(z).stem})
        else:
            # Strategy 2: Flat WAVs
            wavs = [
                f
                for f in self.all_files
                if f.lower().endswith(".wav") and not f.startswith("._")
            ]
            # Chunk them
            for i in range(0, len(wavs), BATCH_SIZE_FILES):
                chunk = wavs[i : i + BATCH_SIZE_FILES]
                batches.append(
                    {
                        "type": "wav_chunk",
                        "targets": chunk,
                        "id": f"batch_{i // BATCH_SIZE_FILES}",
                    }
                )

        return batches

    def process_batch(self, batch_info: dict) -> Generator[Path, None, None]:
        """
        Context manager generator that:
        1. Creates a temp dir for this batch.
        2. Extracts specific files.
        3. Yields the path.
        4. Cleans up files on exit.
        """
        batch_dir = self.temp_root / f"processing_{batch_info['id']}"
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        batch_dir.mkdir()

        try:
            if batch_info["type"] == "nested_zip":
                # Extract the zip file itself
                zip_path_in_archive = batch_info["target"]
                self.zip_ref.extract(zip_path_in_archive, path=batch_dir)

                # Now extract the contents of that nested zip
                nested_zip_path = batch_dir / zip_path_in_archive
                try:
                    with zipfile.ZipFile(nested_zip_path, "r") as nested_ref:
                        nested_ref.extractall(batch_dir)

                    # Remove the nested zip file to save space, keeping extracted contents
                    os.remove(nested_zip_path)
                except zipfile.BadZipFile:
                    st.warning(f"Skipping bad zip: {zip_path_in_archive}")

            elif batch_info["type"] == "wav_chunk":
                # Extract specific wav files
                for f in batch_info["targets"]:
                    self.zip_ref.extract(f, path=batch_dir)

            yield batch_dir

        finally:
            # Cleanup!
            if batch_dir.exists():
                shutil.rmtree(batch_dir)


def parse_script_files(directory: Path) -> Dict[str, str]:
    """
    Parse SCRIPT/*.TXT files to get ground truth text.
    Recursively finds text files.
    """
    scripts = {}
    txt_files = []

    for root, dirs, files in os.walk(directory):
        if "__MACOSX" in root:
            continue
        for file in files:
            if file.upper().endswith(".TXT") and not file.startswith("._"):
                txt_files.append(Path(root) / file)

    for txt_path in txt_files:
        try:
            # Handle BOM and encoding
            content = ""
            try:
                with open(txt_path, "r", encoding="utf-8-sig") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read()

            content = content.lstrip("\ufeff").lstrip()
            lines = content.splitlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue

                # Regex for ID (e.g., 000010001 or just digits)
                match = re.match(r"^(\d+)", line)
                if match:
                    file_id = match.group(1)
                    text = ""

                    parts = line.split("\t", 1)
                    if len(parts) > 1 and parts[1].strip():
                        text = parts[1].strip()
                        i += 1
                    elif i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if not re.match(r"^\d+", next_line.strip()):
                            text = next_line.strip()
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1

                    if file_id and text:
                        scripts[file_id] = text
                else:
                    i += 1
        except Exception as e:
            logger.warning(f"Failed to parse script {txt_path.name}: {e}")

    return scripts


def find_wav_files(directory: Path) -> List[Path]:
    """Recursively find .WAV files."""
    wavs = []
    for root, dirs, files in os.walk(directory):
        if "__MACOSX" in root:
            continue
        for file in files:
            if file.upper().endswith(".WAV") and not file.startswith("._"):
                wavs.append(Path(root) / file)
    return sorted(wavs)


def process_single_file(
    wav_path: Path,
    sentence: str,
    align_model,
    align_metadata,
    phone_model,
    device,
    mfcc_transform,
):
    """
    Runs pipeline on one file:
    1. Load Audio
    2. Forced Alignment
    3. Cut words -> Phoneme Recognition -> MFCC
    """
    try:
        # 1. Load Audio
        wav_numpy = load_audio(str(wav_path), sr=SAMPLE_RATE)
        wav_tensor = torch.from_numpy(wav_numpy).float().to(device)
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        # 2. Forced Alignment
        transcript = [
            {"text": sentence, "start": 0.0, "end": wav_tensor.shape[1] / SAMPLE_RATE}
        ]

        alignment_result = align(
            transcript,
            align_model,
            align_metadata,
            wav_tensor,
            str(device),
            return_char_alignments=False,
        )

        word_segments = alignment_result.get("word_segments", [])
        processed_words = []

        # 3. Word Processing
        for word_seg in word_segments:
            w_start = word_seg["start"]
            w_end = word_seg["end"]
            w_text = word_seg["word"]

            start_sample = int(w_start * SAMPLE_RATE)
            end_sample = int(w_end * SAMPLE_RATE)

            if end_sample - start_sample < MIN_AUDIO_LEN:
                processed_words.append(
                    {
                        "word": w_text,
                        "start": w_start,
                        "end": w_end,
                        "phonemes": "",
                        "mfcc": None,
                        "score": word_seg["score"],
                    }
                )
                continue

            # --- Segment Extraction ---
            segment_wav = wav_tensor[:, start_sample:end_sample]

            # Normalize for Phoneme Model
            mean = segment_wav.mean()
            std = segment_wav.std()
            segment_wav_norm = (segment_wav - mean) / (std + 1e-9)

            with torch.no_grad():
                y_pred, _, _ = phone_model(segment_wav_norm)

            phone_indices, _, _, _ = decode_lattice(y_pred[0].cpu().numpy())
            ipa_symbols = [to_symbol(i) for i in phone_indices]
            ipa_string = " ".join(ipa_symbols)

            # --- MFCC Extraction ---
            # IMPORTANT: Extracted after split, on the specific segment
            mfcc_input = segment_wav
            if mfcc_input.shape[-1] < 400:
                pad_amt = 400 - mfcc_input.shape[-1]
                mfcc_input = torch.nn.functional.pad(mfcc_input, (0, pad_amt))

            mfcc = mfcc_transform(mfcc_input)
            mfcc_list = mfcc.squeeze(0).cpu().numpy().tolist()

            processed_words.append(
                {
                    "word": w_text,
                    "start": w_start,
                    "end": w_end,
                    "phonemes": ipa_string,
                    "mfcc": mfcc_list,
                    "score": word_seg["score"],
                }
            )

        return processed_words, None

    except Exception as e:
        return None, str(e)


def save_results_incrementally(
    new_rows: List[dict], output_dir: Path, processed_speakers: set
):
    """
    Appends results to CSV files immediately.
    Groups by speaker ID.
    """
    if not new_rows:
        return

    df_batch = pd.DataFrame(new_rows)
    # Group by speaker
    for speaker_id, group_df in df_batch.groupby("speaker_id"):
        out_file = output_dir / f"{speaker_id}.csv"

        # Check if we need to write header
        write_header = not out_file.exists()

        # Append to file
        group_df_clean = group_df.drop(columns=["speaker_id"])
        group_df_clean.to_csv(out_file, mode="a", header=write_header, index=False)

        processed_speakers.add(speaker_id)


def main():
    st.set_page_config(
        page_title="Batch Audio Processor", page_icon="🏭", layout="wide"
    )
    st.title("🏭 Large-Scale Word Alignment & Phoneme Recognition")
    st.markdown("""
    **Batch Processing Mode enabled.**
    This tool processes large ZIP archives (e.g., 100GB+) by extracting and processing chunks sequentially
    to save disk space. Results are saved incrementally.
    """)

    # --- Sidebar ---
    st.sidebar.header("Configuration")
    device = get_device()
    st.sidebar.info(f"Device: **{device}**")

    # Input
    st.sidebar.subheader("Input Data")
    input_source = st.sidebar.radio(
        "Source Type",
        ["Local ZIP Path", "Upload ZIP"],
        help="Use 'Local ZIP Path' for large files to avoid browser upload limits.",
    )

    zip_path_str = None
    uploaded_zip = None

    if input_source == "Local ZIP Path":
        zip_path_str = st.sidebar.text_input(
            "Absolute Path to ZIP", value="PART 1 CHANNEL0.zip"
        )
    else:
        uploaded_zip = st.sidebar.file_uploader("Upload ZIP", type=["zip"])

    # Output
    default_out = str(Path(os.getcwd()) / "processed_results")
    custom_out_dir = st.sidebar.text_input(
        "Output Directory",
        value=default_out,
        help="Directory where CSV results will be saved incrementally.",
    )

    # Subset Selection
    st.sidebar.divider()
    st.sidebar.subheader("Subset Selection")

    limit_data = st.sidebar.checkbox("Limit Number of Speakers", value=True)
    n_speakers = 10
    if limit_data:
        n_speakers = st.sidebar.number_input(
            "Max Number of Speakers (Random)", min_value=1, value=10, step=1
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
            "Max Files per Speaker (Random)", min_value=1, value=5, step=1
        )

    # --- Main Logic ---
    if st.button("🚀 Start Batch Processing", type="primary"):
        # 1. Validation
        source_ref = None
        if input_source == "Local ZIP Path":
            if not zip_path_str or not os.path.exists(zip_path_str):
                st.error(f"File not found: {zip_path_str}")
                return
            source_ref = zip_path_str
        else:
            if not uploaded_zip:
                st.error("Please upload a ZIP file.")
                return
            source_ref = uploaded_zip

        output_dir = Path(custom_out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Resume Logic: Identify Processed Clips ---
        processed_file_ids = set()
        with st.spinner("Scanning output directory to resume..."):
            for csv_file in output_dir.glob("*.csv"):
                try:
                    # We need to know which file_ids are done.
                    # The CSV format includes 'file_id'
                    done_df = pd.read_csv(csv_file, usecols=["file_id"])
                    processed_file_ids.update(done_df["file_id"].astype(str).tolist())
                except Exception:
                    pass

        if processed_file_ids:
            st.info(
                f"♻️ Resuming: Found {len(processed_file_ids)} clips already processed. They will be skipped."
            )

        # 2. Load Models
        with st.spinner("Loading models..."):
            align_model, align_meta, phone_model = load_models(str(device))
            if not align_model:
                return

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

        # 3. Initialize Batch Processor
        status_container = st.container()
        log_container = st.container()

        processed_speakers = set()
        total_files_processed = 0

        # Create a persistent temp dir for the whole run (for scripts)
        with tempfile.TemporaryDirectory() as global_temp_str:
            global_temp = Path(global_temp_str)
            processor = ZipBatchProcessor(source_ref, global_temp)

            try:
                with st.spinner(
                    "Scanning archive structure (this may take a moment)..."
                ):
                    # --- Apply Filtering if Requested ---
                    if limit_data:
                        spk_count, clip_count = processor.filter_dataset(
                            n_speakers, n_clips_per_speaker, limit_clips
                        )
                        st.info(
                            f"Filtering applied. Selected {spk_count} speakers and {clip_count} total files."
                        )

                    # Extract Metadata Scripts First
                    scripts = processor.extract_scripts()
                    st.info(f"Loaded {len(scripts)} script entries.")

                    # Identify Batches
                    batches = processor.get_batches()
                    st.success(f"Identified {len(batches)} batches to process.")

                # 4. Batch Loop
                batch_progress = st.progress(0)

                for b_idx, batch in enumerate(batches):
                    status_container.markdown(
                        f"### Processing Batch {b_idx + 1}/{len(batches)}: `{batch['id']}`"
                    )

                    # Process this batch
                    for batch_dir in processor.process_batch(batch):
                        # Update scripts with any found in this specific batch (e.g. inside nested zips)
                        local_scripts = parse_script_files(batch_dir)
                        if local_scripts:
                            scripts.update(local_scripts)

                        # Find audio
                        wav_files = find_wav_files(batch_dir)

                        if not wav_files:
                            log_container.text(
                                f"⚠️ Batch {batch['id']}: No WAV files found."
                            )
                            continue

                        batch_results = []

                        # Process files in this batch
                        for idx, wav_path in enumerate(wav_files):
                            file_id = wav_path.stem

                            # --- Resume Check ---
                            if file_id in processed_file_ids:
                                # Skip processed file
                                continue

                            # Heuristic for text matching
                            sentence = scripts.get(file_id, "")

                            if not sentence:
                                # Try stripping speaker ID prefix if strict match fails
                                # Heuristic: if file is 000010001, maybe script is 000010001
                                log_container.text(f"⚠️ {file_id}: Missing script")
                                continue

                            words_data, error = process_single_file(
                                wav_path,
                                sentence,
                                align_model,
                                align_meta,
                                phone_model,
                                device,
                                mfcc_transform,
                            )

                            if words_data:
                                speaker_id = file_id[:6]  # First 6 digits is speaker

                                for wd in words_data:
                                    batch_results.append(
                                        {
                                            "speaker_id": speaker_id,
                                            "file_id": file_id,
                                            "filename": wav_path.name,
                                            "word": wd["word"],
                                            "start_time": f"{wd['start']:.3f}",
                                            "end_time": f"{wd['end']:.3f}",
                                            "phonemes": wd["phonemes"],
                                            "confidence": f"{wd['score']:.4f}",
                                            "mfcc_json": json.dumps(wd["mfcc"])
                                            if wd["mfcc"]
                                            else "",
                                        }
                                    )
                            else:
                                log_container.text(f"❌ {file_id}: {error}")

                            # --- Memory Cleanup within batch ---
                            if idx % 20 == 0:
                                clear_memory()

                        # Save Batch Results Immediately
                        if batch_results:
                            save_results_incrementally(
                                batch_results, output_dir, processed_speakers
                            )
                            total_files_processed += len(wav_files)
                            log_container.text(
                                f"✅ Batch {batch['id']} done. Saved {len(batch_results)} word rows."
                            )

                    # --- Memory Cleanup after batch ---
                    clear_memory()

                    # Update Progress
                    batch_progress.progress((b_idx + 1) / len(batches))

            finally:
                processor.close()

        st.success(
            f"Processing Complete! Processed {total_files_processed} audio files across {len(processed_speakers)} speakers."
        )
        st.info(f"Results saved to: `{output_dir}`")


if __name__ == "__main__":
    main()
