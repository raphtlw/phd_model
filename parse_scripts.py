import csv
import io
import os
import re
import zipfile
from pathlib import Path


def process_zip_scripts(zip_path, output_csv_path):
    """
    Reads a zip file directly, parses transcript files in the SCRIPT/ folder,
    and outputs a consolidated CSV.
    """

    # Configuration
    target_folder = "SCRIPT/"
    # Regex to match the ID and the Transcript on the first line
    # Matches: Start of line, digits (ID), tab, text
    line1_pattern = re.compile(r"^(\d+)\t(.+)")

    data_rows = []
    seen_ids = set()  # To handle duplicates

    print(f"Reading zip file: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            # Filter for .TXT files in SCRIPT/ directory, ignoring hidden/system files
            file_list = [
                f
                for f in z.namelist()
                if f.startswith(target_folder)
                and f.endswith(".TXT")
                and not f.split("/")[-1].startswith(".")
            ]

            total_files = len(file_list)
            print(f"Found {total_files} script files. Processing...")

            for file_name in file_list:
                # Extract potential metadata from filename (e.g., '002740' from 'SCRIPT/002740.TXT')
                # In NSC Part 1, this filename is often the Speaker ID or Session ID.
                base_name = os.path.basename(file_name)
                file_id = os.path.splitext(base_name)[0]

                with z.open(file_name) as f:
                    # use utf-8-sig to automatically handle the BOM/weird empty char at the start
                    text_wrapper = io.TextIOWrapper(
                        f, encoding="utf-8-sig", newline=None
                    )

                    lines = text_wrapper.readlines()

                    # Iterate through lines to find pairs
                    i = 0
                    while i < len(lines):
                        current_line = lines[i].strip()

                        # skip empty lines
                        if not current_line:
                            i += 1
                            continue

                        # Check if line matches the ID<tab>Transcript format
                        match = line1_pattern.match(current_line)

                        if match:
                            utterance_id = match.group(1)
                            original_transcript = match.group(2).strip()

                            # Try to get the next line for the normalized text
                            normalized_transcript = ""
                            if i + 1 < len(lines):
                                next_line = lines[i + 1].strip()
                                # The second line usually starts with a tab or is just text
                                # We check if it DOESN'T look like a new ID line
                                if next_line and not line1_pattern.match(next_line):
                                    normalized_transcript = next_line.strip()
                                    i += 1  # Skip the next line as we consumed it

                            # Duplicate check
                            if utterance_id not in seen_ids:
                                seen_ids.add(utterance_id)
                                data_rows.append(
                                    {
                                        "Utterance_ID": utterance_id,
                                        "Timestamp_Sort_Key": int(
                                            utterance_id
                                        ),  # Helper for sorting
                                        "File_ID": file_id,  # Likely Speaker ID
                                        "Original_Script": original_transcript,
                                        "Normalized_Script": normalized_transcript,
                                        "Source_Path": file_name,
                                    }
                                )

                        i += 1

    except FileNotFoundError:
        print(f"Error: The file {zip_path} was not found.")
        return
    except zipfile.BadZipFile:
        print(f"Error: The file {zip_path} is not a valid zip file.")
        return

    # Sort by the integer value of the timestamp/ID
    print("Sorting data...")
    data_rows.sort(key=lambda x: x["Timestamp_Sort_Key"])

    # Write to CSV
    print(f"Writing {len(data_rows)} unique lines to {output_csv_path}...")
    headers = [
        "Utterance_ID",
        "File_ID",
        "Original_Script",
        "Normalized_Script",
        "Source_Path",
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in data_rows:
            # Remove the sort key before writing
            del row["Timestamp_Sort_Key"]
            writer.writerow(row)

    print("Done!")


if __name__ == "__main__":
    # Path to your external drive zip file
    zip_file_path = "/Volumes/TOSHIBA EXT/PART 1 CHANNEL0.zip"

    # Output file in the current directory
    output_file = "combined_scripts.csv"

    process_zip_scripts(zip_file_path, output_file)
