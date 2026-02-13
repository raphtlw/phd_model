"""
Streamlit App: Wav2Vec2 Model Accuracy Comparison
Compares phoneme transcription accuracy between full and quantized models.

Usage:
    streamlit run model_comparison_app.py
"""

import io
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

# IPA Symbol Dictionary (from your project)
SYMBOLS = {
    "r": 1,
    "ʝ": 2,
    "ã": 3,
    "gː": 4,
    "t": 5,
    "n": 6,
    "w": 7,
    "u": 8,
    "l": 9,
    "yː": 10,
    "ʎ": 11,
    "bʲ": 12,
    "ə": 13,
    "ʃʲ": 14,
    "sː": 15,
    "zʲ": 16,
    "kː": 17,
    "y": 18,
    "ɒ": 19,
    "fʲ": 20,
    "ɑ": 21,
    "ʏ": 22,
    "ɣ": 23,
    "s": 24,
    "m": 25,
    "tː": 26,
    "xʲ": 27,
    "vː": 28,
    "ø": 29,
    "h": 30,
    "ɨ": 31,
    "dʲ": 32,
    "dː": 33,
    "bː": 34,
    "ɲː": 35,
    "ɑː": 36,
    "ɪ": 37,
    "ɛ": 38,
    "i": 39,
    "ʔ": 40,
    "g": 41,
    "ʃ": 42,
    "ɜː": 43,
    "mː": 44,
    "øː": 45,
    "fː": 46,
    "p": 47,
    "iː": 48,
    "(...)": 49,
    "v": 50,
    "ʌ": 51,
    "b": 52,
    "k": 53,
    "x": 54,
    "ɲ": 55,
    "ʒ": 56,
    "rː": 57,
    "eː": 58,
    "ç": 59,
    "ŋ": 60,
    "ɔː": 61,
    "œ": 62,
    "ẽ": 63,
    "θ": 64,
    "a": 65,
    "rʲ": 66,
    "vʲ": 67,
    "ʃː": 68,
    "æ": 69,
    "ɶ̃": 70,
    "pː": 71,
    "nː": 72,
    "lʲ": 73,
    "õ": 74,
    "pʲ": 75,
    "ɱ": 76,
    "ð": 77,
    "f": 78,
    "j": 79,
    "o": 80,
    "nʲ": 81,
    "sʲ": 82,
    "lː": 83,
    "e": 84,
    "d": 85,
    "ʊ": 86,
    "gʲ": 87,
    "z": 88,
    "ɛː": 89,
    "tʲ": 90,
    "β": 91,
    "mʲ": 92,
    "uː": 93,
    "ɥ": 94,
    "ʀ": 95,
    "aː": 96,
    "ɐ": 97,
    "ɔ": 98,
    "oː": 99,
    "ʎː": 100,
    "kʲ": 101,
}

INDEX_TO_SYMBOL = {v: k for k, v in SYMBOLS.items()}


class Wav2Vec2Transcriber:
    """ONNX-based Wav2Vec2 phoneme transcriber"""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess_audio(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        """Normalize audio to mean=0, std=1"""
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        if sample_rate != 16000:
            st.warning(
                f"Audio sample rate {sample_rate} != 16000. Accuracy may be affected."
            )

        mean = np.mean(audio)
        std = np.std(audio)
        normalized = (audio - mean) / (std + 1e-9)

        return normalized.astype(np.float32)

    def decode_ctc(self, logits: np.ndarray) -> str:
        """CTC greedy decoding"""
        raw_indices = np.argmax(logits[0], axis=1)

        phones = []
        prev_idx = -1

        for idx in raw_indices:
            if idx != prev_idx:
                if idx != 0:
                    symbol = INDEX_TO_SYMBOL.get(idx, "")
                    if symbol:
                        phones.append(symbol)
                prev_idx = idx

        return " ".join(phones)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to IPA"""
        audio, sr = sf.read(audio_path)
        processed = self.preprocess_audio(audio, sr)
        tensor = processed.reshape(1, -1)

        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0]

        return self.decode_ctc(logits)


def parse_ground_truth_txt(txt_path: Path) -> Dict[str, str]:
    """
    Parse ground truth .TXT file with format:
    filename    original_text
                actual_transcription

    Returns dict: {filename: actual_transcription}
    """
    ground_truth = {}

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Check if line starts with a filename (digits)
        if re.match(r"^\d+", line):
            parts = line.split("\t", 1)
            if len(parts) >= 1:
                file_id = parts[0].strip()

                # Next line should contain the actual transcription
                if i + 1 < len(lines):
                    actual_text = lines[i + 1].strip()
                    # Remove leading tab if present
                    if actual_text.startswith("\t"):
                        actual_text = actual_text[1:]

                    ground_truth[file_id] = actual_text
                    i += 2  # Skip both lines
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    return ground_truth


def calculate_similarity(str1: str, str2: str) -> Dict[str, float]:
    """Calculate similarity between two IPA strings using Levenshtein distance"""
    tokens1 = str1.split()
    tokens2 = str2.split()

    # Calculate edit distance
    m, n = len(tokens1), len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    edit_distance = dp[m][n]

    # Calculate similarity percentage
    max_len = max(m, n)
    similarity = (1 - edit_distance / max_len) * 100 if max_len > 0 else 100

    # Calculate Phone Error Rate
    per = (edit_distance / n) * 100 if n > 0 else 0

    return {
        "similarity": similarity,
        "per": per,
        "edit_distance": edit_distance,
        "length1": m,
        "length2": n,
    }


def extract_audio_files(uploaded_file) -> Path:
    """Extract zip file to temp directory"""
    temp_dir = Path(tempfile.mkdtemp())

    if uploaded_file.name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as zf:
            zf.extractall(temp_dir)
    else:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

    return temp_dir


def find_wav_files(directory: Path) -> List[Path]:
    """Recursively find all .WAV files, excluding macOS metadata files"""
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Skip macOS metadata files and hidden files
            if file.startswith("._") or file.startswith("."):
                continue
            if file.upper().endswith(".WAV"):
                wav_files.append(Path(root) / file)
    return sorted(wav_files)


def find_ground_truth_files(directory: Path) -> List[Path]:
    """Find all .TXT ground truth files in the extracted directory"""
    txt_files = []
    for root, dirs, files in os.walk(directory):
        if "__MACOSX" in root:
            continue
        for file in files:
            if file.startswith("._") or file.startswith("."):
                continue
            if file.upper().endswith(".TXT"):
                txt_files.append(Path(root) / file)
    return sorted(txt_files)


def main():
    st.set_page_config(
        page_title="Wav2Vec2 Model Comparison", page_icon="🎙️", layout="wide"
    )

    st.title("🎙️ Wav2Vec2 Model Accuracy Comparison")
    st.markdown(
        "Compare phoneme transcription accuracy between **full** and **quantized** models"
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Model source selection
    st.sidebar.subheader("Model Files")
    model_source = st.sidebar.radio(
        "Model Source",
        ["Local Path", "Upload Files"],
        help="Choose to use local model files or upload them",
    )

    full_model = None
    quant_model = None
    full_model_path = None
    quant_model_path = None

    if model_source == "Local Path":
        full_model_path = st.sidebar.text_input(
            "Full Model Path",
            value="./wav2vec2_commonphone.onnx",
            help="Path to the full ONNX model",
        )
        quant_model_path = st.sidebar.text_input(
            "Quantized Model Path",
            value="./wav2vec2_commonphone.quant.onnx",
            help="Path to the quantized ONNX model",
        )

        # Validate paths
        if full_model_path and not os.path.exists(full_model_path):
            st.sidebar.error(f"Full model not found: {full_model_path}")
        elif full_model_path:
            st.sidebar.success(f"✓ Full model found")

        if quant_model_path and not os.path.exists(quant_model_path):
            st.sidebar.error(f"Quantized model not found: {quant_model_path}")
        elif quant_model_path:
            st.sidebar.success(f"✓ Quantized model found")
    else:
        full_model = st.sidebar.file_uploader(
            "Upload Full Model (.onnx)", type=["onnx"], key="full_model"
        )

        quant_model = st.sidebar.file_uploader(
            "Upload Quantized Model (.onnx)", type=["onnx"], key="quant_model"
        )

    # Audio files upload
    st.sidebar.subheader("Audio Files")
    audio_upload = st.sidebar.file_uploader(
        "Upload Audio Files (.zip - should contain .TXT ground truth)",
        type=["zip"],
        key="audio_files",
        help="Upload SPEAKER0128.zip containing SESSION folders and .TXT files",
    )

    # Process button
    if st.sidebar.button("🚀 Run Comparison", type="primary"):
        # Validate inputs
        if model_source == "Local Path":
            if not (full_model_path and quant_model_path and audio_upload):
                st.error("Please provide model paths and audio files")
                return
            if not os.path.exists(full_model_path):
                st.error(f"Full model not found: {full_model_path}")
                return
            if not os.path.exists(quant_model_path):
                st.error(f"Quantized model not found: {quant_model_path}")
                return
        else:
            if not (full_model and quant_model and audio_upload):
                st.error("Please upload both models and audio files")
                return

            # Save uploaded models to temp files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as f:
                f.write(full_model.read())
                full_model_path = f.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as f:
                f.write(quant_model.read())
                quant_model_path = f.name

        # Extract audio files
        with st.spinner("Extracting audio files..."):
            audio_dir = extract_audio_files(audio_upload)
            wav_files = find_wav_files(audio_dir)
            gt_files = find_ground_truth_files(audio_dir)

        st.success(f"✅ Found {len(wav_files)} WAV files")

        # Parse ground truth
        ground_truth = {}
        if gt_files:
            st.success(f"✅ Found {len(gt_files)} ground truth files")
            for gt_file in gt_files:
                try:
                    file_gt = parse_ground_truth_txt(gt_file)
                    ground_truth.update(file_gt)
                except Exception as e:
                    st.warning(f"Error parsing {gt_file.name}: {e}")

            st.info(f"📋 Loaded {len(ground_truth)} ground truth transcriptions")
        else:
            st.warning(
                "⚠️ No .TXT ground truth files found. Results will not include accuracy metrics."
            )

        # Initialize transcribers
        with st.spinner("Loading models..."):
            full_transcriber = Wav2Vec2Transcriber(full_model_path)
            quant_transcriber = Wav2Vec2Transcriber(quant_model_path)

        # Process files
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, wav_file in enumerate(wav_files):
            file_id = wav_file.stem
            status_text.text(f"Processing {idx + 1}/{len(wav_files)}: {file_id}")

            try:
                # Transcribe with both models
                full_ipa = full_transcriber.transcribe(str(wav_file))
                quant_ipa = quant_transcriber.transcribe(str(wav_file))

                # Compare models
                comparison = calculate_similarity(full_ipa, quant_ipa)

                result = {
                    "file_id": file_id,
                    "file_path": str(wav_file.relative_to(audio_dir)),
                    "full_model_ipa": full_ipa,
                    "quant_model_ipa": quant_ipa,
                    "model_similarity": comparison["similarity"],
                    "model_per": comparison["per"],
                    "edit_distance": comparison["edit_distance"],
                }

                # Calculate accuracy if ground truth available
                if file_id in ground_truth:
                    gt_text = ground_truth[file_id]
                    result["ground_truth_text"] = gt_text
                    result["has_ground_truth"] = True
                else:
                    result["has_ground_truth"] = False

                results.append(result)

            except Exception as e:
                st.warning(f"Error processing {file_id}: {str(e)}")

            progress_bar.progress((idx + 1) / len(wav_files))

        status_text.text("✅ Processing complete!")

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Display results
        st.header("📊 Results")

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Files Processed", len(df))

        with col2:
            gt_count = df["has_ground_truth"].sum()
            st.metric("Files with Ground Truth", gt_count)

        with col3:
            avg_similarity = df["model_similarity"].mean()
            st.metric(
                "Avg Model Similarity",
                f"{avg_similarity:.2f}%",
                help="Average similarity between full and quantized model outputs",
            )

        with col4:
            avg_per = df["model_per"].mean()
            st.metric(
                "Avg Phone Error Rate",
                f"{avg_per:.2f}%",
                help="Average PER of quantized vs full model",
            )

        # Model comparison visualization
        st.subheader("📈 Model Comparison Statistics")

        col1, col2 = st.columns(2)

        with col1:
            # Similarity distribution
            fig_hist = px.histogram(
                df,
                x="model_similarity",
                nbins=20,
                title="Model Similarity Distribution",
                labels={"model_similarity": "Similarity (%)"},
            )
            fig_hist.update_layout(
                xaxis_title="Similarity (%)",
                yaxis_title="Count",
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # PER distribution
            fig_per = px.histogram(
                df,
                x="model_per",
                nbins=20,
                title="Phone Error Rate Distribution",
                labels={"model_per": "PER (%)"},
            )
            fig_per.update_layout(
                xaxis_title="Phone Error Rate (%)",
                yaxis_title="Count",
                showlegend=False,
            )
            st.plotly_chart(fig_per, use_container_width=True)

        # Statistics table
        st.subheader("📊 Model Comparison Summary")
        stats_data = {
            "Metric": [
                "Average Similarity",
                "Median Similarity",
                "Min Similarity",
                "Max Similarity",
                "Std Dev",
                "Average PER",
                "Median PER",
            ],
            "Value": [
                f"{df['model_similarity'].mean():.2f}%",
                f"{df['model_similarity'].median():.2f}%",
                f"{df['model_similarity'].min():.2f}%",
                f"{df['model_similarity'].max():.2f}%",
                f"{df['model_similarity'].std():.2f}%",
                f"{df['model_per'].mean():.2f}%",
                f"{df['model_per'].median():.2f}%",
            ],
        }
        st.table(pd.DataFrame(stats_data))

        # Comparison samples
        st.subheader("🔍 Sample Transcriptions")

        # Show files with lowest similarity (potential issues)
        st.write("**Files with Lowest Model Agreement:**")
        low_similarity = df.nsmallest(5, "model_similarity")[
            [
                "file_id",
                "model_similarity",
                "model_per",
                "full_model_ipa",
                "quant_model_ipa",
            ]
        ]
        st.dataframe(low_similarity, use_container_width=True)

        # Show files with highest similarity
        st.write("**Files with Highest Model Agreement:**")
        high_similarity = df.nlargest(5, "model_similarity")[
            [
                "file_id",
                "model_similarity",
                "model_per",
                "full_model_ipa",
                "quant_model_ipa",
            ]
        ]
        st.dataframe(high_similarity, use_container_width=True)

        # Side-by-side comparison
        st.subheader("📝 Detailed Comparison Examples")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Full Model Output")
            for idx, row in df.head(5).iterrows():
                with st.expander(
                    f"{row['file_id']} (Similarity: {row['model_similarity']:.1f}%)"
                ):
                    st.text(f"IPA: {row['full_model_ipa']}")
                    if row["has_ground_truth"]:
                        st.text(f"GT: {row['ground_truth_text']}")

        with col2:
            st.markdown("#### Quantized Model Output")
            for idx, row in df.head(5).iterrows():
                with st.expander(f"{row['file_id']} (PER: {row['model_per']:.1f}%)"):
                    st.text(f"IPA: {row['quant_model_ipa']}")
                    st.text(f"Edit Dist: {row['edit_distance']}")

        # Full results table
        st.subheader("📋 All Results")
        st.dataframe(df, use_container_width=True)

        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="model_comparison_results.csv",
            mime="text/csv",
        )

        # Cleanup temp files only if we created them
        if model_source == "Upload Files":
            try:
                os.unlink(full_model_path)
                os.unlink(quant_model_path)
            except:
                pass


if __name__ == "__main__":
    main()
