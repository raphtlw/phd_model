import fnmatch
import io
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="IMDA Script Search", page_icon="🔍", layout="wide")

# --- Parsing Logic ---


def parse_single_script_content(content: str, filename: str) -> List[Dict]:
    """
    Parses the raw text content of an IMDA script file.
    Returns a list of dictionaries: {'file_id': str, 'text': str, 'filename': str}
    """
    entries = []
    lines = content.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # IMDA IDs are typically digits (e.g., 000010001)
        # Regex looks for line starting with digits
        match = re.match(r"^(\d+)", line)
        if match:
            file_id = match.group(1)
            text = ""

            # Strategy 1: Tab separated on same line
            parts = line.split("\t", 1)
            if len(parts) > 1 and parts[1].strip():
                text = parts[1].strip()
                i += 1
            # Strategy 2: Text is on the next line
            elif i + 1 < len(lines):
                next_line = lines[i + 1]
                # Check if next line is NOT another ID
                if not re.match(r"^\d+", next_line.strip()):
                    text = next_line.strip()
                    i += 2
                else:
                    # Current line is just an ID with no text? Move on.
                    i += 1
            else:
                i += 1

            if file_id and text:
                entries.append(
                    {
                        "file_id": file_id,
                        "text": text,
                        "filename": filename,
                        "speaker_id": file_id[
                            :6
                        ],  # Heuristic: First 6 digits are speaker ID
                    }
                )
        else:
            i += 1

    return entries


# --- Audio Retrieval Logic ---


class AudioFinder:
    """
    Locates and retrieves audio bytes for a specific file_id.
    Handles local directories, simple ZIPs, and nested ZIPs (e.g. Part1.zip -> WAVE/SPEAKER123.zip).
    """

    @staticmethod
    def get_audio(
        source: Union[str, Path, io.BytesIO],
        source_type: str,
        file_id: str,
        speaker_id: str,
    ) -> Optional[bytes]:
        # IMDA files are typically .wav
        wav_filename = f"{file_id}.wav"

        try:
            if source_type == "Local Directory":
                return AudioFinder._find_in_directory(Path(source), wav_filename)

            elif source_type in ["Local ZIP Path", "Upload ZIP"]:
                return AudioFinder._find_in_zip(source, wav_filename, speaker_id)

        except Exception as e:
            st.error(f"Error fetching audio: {e}")
            return None
        return None

    @staticmethod
    def _find_in_directory(root_dir: Path, target_name: str) -> Optional[bytes]:
        # Walk directory to find the file (case-insensitive)
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.lower() == target_name.lower():
                    try:
                        with open(os.path.join(root, f), "rb") as wav_file:
                            return wav_file.read()
                    except Exception:
                        return None
        return None

    @staticmethod
    def _find_in_zip(
        source: Union[str, Path, io.BytesIO], target_name: str, speaker_id: str
    ) -> Optional[bytes]:
        # Prepare the zip file object
        context_manager = None
        if isinstance(source, (str, Path)):
            context_manager = zipfile.ZipFile(source, "r")
        else:
            # Reset pointer for uploaded files
            source.seek(0)
            context_manager = zipfile.ZipFile(source, "r")

        with context_manager as zf:
            all_files = zf.namelist()

            # --- 1. Check Root/Flat Level (Fastest) ---
            # Finds 'Audio/000010001.wav' or just '000010001.wav'
            exact_matches = [
                f
                for f in all_files
                if f.lower().endswith(target_name.lower()) and "__MACOSX" not in f
            ]
            if exact_matches:
                return zf.read(exact_matches[0])

            # --- 2. Check Nested Zips (IMDA Structure) ---
            # Identify candidate nested zips.
            # Priority A: Zips that contain the speaker ID (e.g., SPEAKER000010.zip)
            # Priority B: All other Zips in a 'WAVE' folder (Fallback)

            nested_zips = [
                f
                for f in all_files
                if f.lower().endswith(".zip") and "__MACOSX" not in f
            ]

            # Filter for likely candidates to save time
            primary_candidates = [nz for nz in nested_zips if speaker_id in nz]
            secondary_candidates = [
                nz
                for nz in nested_zips
                if "wave" in nz.lower() and nz not in primary_candidates
            ]

            # Combine lists, primary first
            search_queue = primary_candidates + secondary_candidates

            for nz_name in search_queue:
                try:
                    # Read the nested zip into memory
                    nested_zip_bytes = io.BytesIO(zf.read(nz_name))
                    with zipfile.ZipFile(nested_zip_bytes) as nested_zf:
                        nested_files = nested_zf.namelist()

                        # Look for the wav inside this nested zip
                        wav_matches = [
                            wf
                            for wf in nested_files
                            if wf.lower().endswith(target_name.lower())
                        ]
                        if wav_matches:
                            return nested_zf.read(wav_matches[0])
                except zipfile.BadZipFile:
                    continue
                except Exception:
                    continue

        return None


# --- Ingestion Logic (Recursive Zip Handling) ---


class RecursiveScriptProcessor:
    """
    Handles deep processing of zip files without full extraction.
    extracts nested zips to temp, processes them, and cleans up.
    """

    def __init__(self):
        self.all_entries = []
        self.status_text = st.empty()
        self.file_counter = 0

    def process_source(self, source: Union[str, Path, io.BytesIO]):
        """Entry point for processing a zip source."""
        self.all_entries = []
        self.file_counter = 0

        try:
            # Handle uploaded file needing reset
            if hasattr(source, "seek"):
                source.seek(0)

            if isinstance(source, (str, Path)):
                if not os.path.exists(source):
                    st.error(f"File not found: {source}")
                    return pd.DataFrame()

            # Open the main zip
            with zipfile.ZipFile(source, "r") as zf:
                self._scan_zip(zf, parent_name="Root")

        except zipfile.BadZipFile:
            st.error("Invalid or Corrupt ZIP file.")
        except Exception as e:
            st.error(f"Error processing zip: {e}")

        return pd.DataFrame(self.all_entries)

    def _scan_zip(self, zf_obj: zipfile.ZipFile, parent_name: str):
        """Scans a ZipFile object only for .txt files."""
        file_list = zf_obj.namelist()

        # Filter for relevant files to avoid iterating uselessly
        txt_files = [
            f
            for f in file_list
            if f.lower().endswith(".txt")
            and not f.startswith("._")
            and "__MACOSX" not in f
        ]

        # Process Text Files in current level
        for filename in txt_files:
            try:
                with zf_obj.open(filename) as f:
                    content_bytes = f.read()
                    content = self._decode(content_bytes)
                    entries = parse_single_script_content(content, Path(filename).name)
                    self.all_entries.extend(entries)

                    self.file_counter += 1
                    if self.file_counter % 50 == 0:
                        self.status_text.text(
                            f"Processed {self.file_counter} text files... (Current: {filename})"
                        )
            except Exception as e:
                # Log but continue
                print(f"Error reading {filename}: {e}")

    def _decode(self, bytes_data: bytes) -> str:
        """Robust decoding helper."""
        try:
            return bytes_data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return bytes_data.decode("utf-8", errors="replace")


def ingest_from_directory(directory: Path) -> pd.DataFrame:
    """Recursively reads all .TXT files from a local directory."""
    all_entries = []
    status_text = st.empty()

    # Find files
    txt_files = []
    for root, dirs, files in os.walk(directory):
        if "__MACOSX" in root:
            continue
        for file in files:
            if file.lower().endswith(".txt") and not file.startswith("._"):
                txt_files.append(Path(root) / file)

    total_files = len(txt_files)
    status_text.text(f"Found {total_files} text files. Parsing...")

    progress_bar = st.progress(0)

    for idx, txt_path in enumerate(txt_files):
        try:
            # Handle encoding (BOM or standard)
            content = ""
            try:
                with open(txt_path, "r", encoding="utf-8-sig") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read()

            entries = parse_single_script_content(content, txt_path.name)
            all_entries.extend(entries)

        except Exception as e:
            pass

        if idx % 10 == 0:
            progress_bar.progress(min((idx + 1) / total_files, 1.0))

    progress_bar.empty()
    status_text.empty()

    return pd.DataFrame(all_entries)


# --- UI Layout ---


def main():
    st.title("🔍 IMDA Dataset Script Ingestor")
    st.markdown("""
    Ingest text scripts from the **IMDA National Speech Corpus**.
    **Now with Audio Playback**: Plays recordings directly from local directories or ZIP archives (including nested speaker zips).
    """)

    # --- Sidebar: Data Loading ---
    with st.sidebar:
        st.header("1. Load Data")
        data_source_type = st.radio(
            "Source", ["Local ZIP Path", "Local Directory", "Upload ZIP"]
        )

        # Check session state for existing data
        if "script_df" in st.session_state:
            st.success(f"✅ Loaded {len(st.session_state.script_df):,} lines")
            st.caption(f"Source: {st.session_state.get('data_source_info', 'Unknown')}")

            if st.button("Clear Data", type="secondary"):
                del st.session_state.script_df
                if "data_source_obj" in st.session_state:
                    del st.session_state.data_source_obj
                if "data_source_type" in st.session_state:
                    del st.session_state.data_source_type
                st.rerun()

        st.divider()

        if "script_df" not in st.session_state:
            if data_source_type == "Local ZIP Path":
                st.info("Best for large datasets (e.g. >1GB) to avoid browser limits.")
                local_zip_path = st.text_input(
                    "Absolute Path to ZIP", value="IMDA_PART1.zip"
                )
                if st.button("Ingest Local ZIP", type="primary"):
                    if os.path.exists(local_zip_path):
                        processor = RecursiveScriptProcessor()
                        with st.spinner("Scanning ZIP..."):
                            df = processor.process_source(local_zip_path)
                            if not df.empty:
                                st.session_state.script_df = df
                                st.session_state.data_source_obj = local_zip_path
                                st.session_state.data_source_type = "Local ZIP Path"
                                st.session_state.data_source_info = local_zip_path
                                st.rerun()
                            else:
                                st.warning("No script data found.")
                    else:
                        st.error("File not found.")

            elif data_source_type == "Local Directory":
                local_path = st.text_input("Directory Path", value=os.getcwd())
                if st.button("Ingest Directory", type="primary"):
                    if os.path.isdir(local_path):
                        with st.spinner("Ingesting..."):
                            df = ingest_from_directory(Path(local_path))
                            if not df.empty:
                                st.session_state.script_df = df
                                st.session_state.data_source_obj = local_path
                                st.session_state.data_source_type = "Local Directory"
                                st.session_state.data_source_info = local_path
                                st.rerun()
                            else:
                                st.warning("No script data found.")
                    else:
                        st.error("Invalid directory path.")

            else:  # Upload ZIP
                uploaded_file = st.file_uploader("Upload Scripts ZIP", type="zip")
                if uploaded_file and st.button("Ingest Uploaded ZIP", type="primary"):
                    processor = RecursiveScriptProcessor()
                    with st.spinner("Reading Uploaded ZIP..."):
                        df = processor.process_source(uploaded_file)
                        if not df.empty:
                            st.session_state.script_df = df
                            st.session_state.data_source_obj = uploaded_file
                            st.session_state.data_source_type = "Upload ZIP"
                            st.session_state.data_source_info = uploaded_file.name
                            st.rerun()
                        else:
                            st.warning("No script data found.")

    # --- Main Area: Search ---

    if "script_df" in st.session_state:
        df = st.session_state.script_df

        # Statistics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Utterances", f"{len(df):,}")
        c2.metric("Unique Files", f"{df['filename'].nunique():,}")
        c3.metric("Unique Speakers", f"{df['speaker_id'].nunique():,}")

        st.divider()

        # Search Controls
        col_search, col_opts = st.columns([3, 1])

        with col_search:
            search_query = st.text_input(
                "Search Query",
                placeholder="Enter word, phrase, or ID...",
                key="search_box",
            )

        with col_opts:
            st.caption("Search Options")
            use_regex = st.checkbox("Use Regex", value=False)
            case_sensitive = st.checkbox("Case Sensitive", value=False)
            search_col = st.selectbox("Search In", ["text", "file_id", "speaker_id"])

        # Filtering Logic
        results = df

        if search_query:
            try:
                if use_regex:
                    results = df[
                        df[search_col]
                        .astype(str)
                        .str.contains(
                            search_query, case=case_sensitive, regex=True, na=False
                        )
                    ]
                else:
                    results = df[
                        df[search_col]
                        .astype(str)
                        .str.contains(
                            search_query, case=case_sensitive, regex=False, na=False
                        )
                    ]
            except Exception as e:
                st.error(f"Invalid Regex: {e}")
                results = pd.DataFrame()

        # Display Results
        st.subheader(f"Results ({len(results)})")

        if not results.empty:
            # --- Audio Playback Section ---
            st.markdown("### 🎧 Play Audio")

            # Create a selection box populated with the current search results
            # Format: "FileID: Text snippet..."
            options = results.apply(
                lambda row: f"{row['file_id']} | {row['text'][:60]}...", axis=1
            ).tolist()
            selected_option = st.selectbox(
                "Select an utterance to listen:", options=options, index=0
            )

            if selected_option:
                # Extract file_id from selection string
                selected_file_id = selected_option.split(" | ")[0]

                # Get full row data for context
                row_data = results[results["file_id"] == selected_file_id].iloc[0]

                c_audio, c_meta = st.columns([1, 2])

                with c_audio:
                    # Fetch Audio Button/Logic
                    # We check if we have the source available
                    if "data_source_obj" in st.session_state:
                        with st.spinner(f"Searching for {selected_file_id}.wav..."):
                            audio_bytes = AudioFinder.get_audio(
                                source=st.session_state.data_source_obj,
                                source_type=st.session_state.data_source_type,
                                file_id=selected_file_id,
                                speaker_id=row_data["speaker_id"],
                            )

                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/wav")
                        else:
                            st.warning("Audio file not found in source.")
                            st.caption(
                                "Tip: Ensure the WAV files are inside the ZIP/Directory (nested Speaker ZIPs are supported)."
                            )
                    else:
                        st.error("Data source lost. Please reload.")

                with c_meta:
                    st.info(
                        f"**Speaker:** {row_data['speaker_id']}\n\n**Full Text:** {row_data['text']}"
                    )

            st.divider()

            # --- Data Table ---
            # Pagination / Limiting for display performance
            max_display = 1000
            if len(results) > max_display:
                st.caption(f"Showing first {max_display} rows.")

            # Helper to make columns nicer
            display_df = results.head(max_display)[
                ["file_id", "text", "speaker_id", "filename"]
            ]

            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "file_id": st.column_config.TextColumn(
                        "Utterance ID", width="small"
                    ),
                    "text": st.column_config.TextColumn("Transcript", width="large"),
                    "speaker_id": st.column_config.TextColumn("Speaker", width="small"),
                    "filename": st.column_config.TextColumn(
                        "Source File", width="medium"
                    ),
                },
            )

            # Export
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Search Results CSV",
                data=csv,
                file_name="imda_search_results.csv",
                mime="text/csv",
            )
        else:
            if search_query:
                st.info("No matches found.")
            else:
                st.info("Enter a search query above.")

    else:
        st.info("👈 Please load data using the sidebar to begin.")


if __name__ == "__main__":
    main()
