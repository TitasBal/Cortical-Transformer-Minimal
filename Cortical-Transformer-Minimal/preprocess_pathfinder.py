# preprocess_pathfinder.py
import os
import numpy as np
import traceback
import time
from PIL import Image # For loading images
import random # Not strictly used, but good to import if ever needed for shuffles

"""
Script to preprocess LRA Pathfinder dataset.
Reads images and metadata from the LRA release structure,
flattens images, pads/truncates them, and saves them as .npy files
along with their labels for easier loading in PyTorch.
"""

# --- Configuration for Preprocessing ---
# Resolutions you want to process from your LRA release data
PATHFINDER_RESOLUTIONS_TO_PROCESS = [32] # Example: [32, 64, 128]. Start with one.

# This maps your desired output split names (e.g., 'train', 'validation', 'test')
# to the LRA difficulty pattern folder names found within each pathfinder<RESOLUTION> directory.
# Ensure these LRA difficulty folder names (e.g., 'curv_baseline') exist.
OUTPUT_SPLIT_MAPPING = {
    'train': 'curv_baseline',
    'validation': 'curv_contour_length_9',
    'test': 'curv_contour_length_14'
}

# **IMPORTANT**: Update these paths to match your system!
# Path to your LRA release folder (the one containing pathfinder32, pathfinder64, etc.)
# Based on your feedback with three 'lra_release' levels:
LRA_RELEASE_ROOT: str = './lra_release/lra_release/lra_release/'
# Path where the processed .npy files will be saved
OUTPUT_PROCESSED_DIR_BASE: str = './lra_pathfinder_preprocessed/'

# Padding ID for pixel sequences
PAD_ID: int = 0 # Assuming 0 is a safe padding value (pixel values 0-255)
# --- End Configuration ---

def load_and_parse_lra_metadata_file(metadata_file_path: str) -> list:
    """
    Loads and parses a single metadata file for Pathfinder.
    Assumes the file is plain text (despite .npy extension), where each line is space-separated.
    Example line from inspect_metadata.py: 'imgs/0 sample_0.png 0 0 1.0 6 2 2 0.5 1 1'
    We need:
        parts[0]: "imgs/0" (first part of relative image path)
        parts[1]: "sample_0.png" (second part of relative image path)
        parts[3]: "0" or "1" (the label)
    """
    print(f"    Attempting to load and parse metadata file: {os.path.basename(metadata_file_path)}")
    parsed_entries_data = [] # List to store dicts of {'image_rel_path': str, 'label': int}
    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            lines_to_parse = f.readlines()

        if not lines_to_parse:
            print(f"    Warning: Metadata file {os.path.basename(metadata_file_path)} is empty.")
            return [] # Return empty list, not None

        for line_idx, line_content in enumerate(lines_to_parse):
            line_content = line_content.strip()
            if not line_content: continue # Skip empty lines
            parts = line_content.split(' ')
            if len(parts) >= 4: # Need at least 4 parts based on inspected format
                try:
                    # parts[0] = "imgs/0", parts[1] = "sample_0.png"
                    relative_image_path = os.path.join(parts[0], parts[1])
                    label = int(parts[3]) # Label is the 4th element (index 3)
                    parsed_entries_data.append({'image_rel_path': relative_image_path, 'label': label})
                except ValueError:
                    # print(f"    Skipping line {line_idx+1} in {os.path.basename(metadata_file_path)} due to non-integer label: '{parts[3]}'")
                    pass # Silently skip lines with non-integer labels for now
                except IndexError:
                    # print(f"    Skipping line {line_idx+1} in {os.path.basename(metadata_file_path)} due to insufficient parts: {parts}")
                    pass
            # else:
                # print(f"    Skipping malformed line {line_idx+1} in {os.path.basename(metadata_file_path)}: '{line_content}'")
        print(f"    Successfully parsed {len(parsed_entries_data)} valid entries from {os.path.basename(metadata_file_path)}.")
        return parsed_entries_data

    except FileNotFoundError: print(f"    ERROR: Metadata file not found: {metadata_file_path}"); return []
    except Exception as e: print(f"    Error loading/parsing metadata file {metadata_file_path}: {e}"); traceback.print_exc(); return []

def process_single_pathfinder_config(resolution: int, source_difficulty_pattern: str, output_split_name_label: str):
    print(f"\nProcessing Pathfinder-{resolution} - Source: '{source_difficulty_pattern}' -> Output: '{output_split_name_label}'...")
    lra_seq_len = resolution * resolution # Assuming single channel for Pathfinder images

    # Path to the specific difficulty folder, e.g., .../lra_release/pathfinder32/curv_baseline/
    source_difficulty_base_path = os.path.join(LRA_RELEASE_ROOT, f"pathfinder{resolution}", source_difficulty_pattern)
    metadata_dir = os.path.join(source_difficulty_base_path, "metadata")
    # Images are located relative to source_difficulty_base_path, according to metadata

    print(f"  Expecting metadata in: {os.path.abspath(metadata_dir)}")
    print(f"  Image paths in metadata will be relative to: {os.path.abspath(source_difficulty_base_path)}")

    if not os.path.isdir(metadata_dir):
        print(f"ERROR: Metadata directory not found: {metadata_dir} (for {resolution}/{source_difficulty_pattern}). Skipping this config.")
        return

    all_X_for_this_output_split, all_Y_for_this_output_split, images_processed_count, entries_skipped_count = [], [], 0, 0
    
    metadata_files = sorted([f for f in os.listdir(metadata_dir) if f.endswith('.npy')])
    if not metadata_files:
        print(f"Warning: No .npy metadata files found in {metadata_dir}. Skipping this config.")
        return

    for mf_name in metadata_files:
        parsed_meta_entries = load_and_parse_lra_metadata_file(os.path.join(metadata_dir, mf_name))
        if not parsed_meta_entries:
            print(f"  No valid entries parsed from {mf_name}. Skipping file.")
            continue

        for entry_data in parsed_meta_entries:
            try:
                relative_image_path = entry_data['image_rel_path'] # e.g., "imgs/0/sample_0.png"
                label = entry_data['label']

                # Construct full path to image.
                # The relative_image_path from metadata is relative to the source_difficulty_base_path
                image_full_path = os.path.join(source_difficulty_base_path, relative_image_path)

                if not os.path.exists(image_full_path):
                    # print(f"  Image not found: {image_full_path}, skipping.")
                    entries_skipped_count +=1
                    continue

                img = Image.open(image_full_path).convert('L') # Convert to grayscale (single channel)
                img_np = np.array(img, dtype=np.int64)       # Pixel values 0-255, cast to int64 for PyTorch
                
                if img_np.shape != (resolution, resolution):
                    # print(f"  Image {image_full_path} has shape {img_np.shape}, expected ({resolution},{resolution}). Skipping.")
                    entries_skipped_count +=1
                    continue

                img_flattened = img_np.reshape(-1) # Flatten to (Res*Res)

                current_len = len(img_flattened)
                if current_len > lra_seq_len:
                    img_flattened = img_flattened[:lra_seq_len]
                elif current_len < lra_seq_len:
                    padding = np.full(lra_seq_len - current_len, PAD_ID, dtype=np.int64)
                    img_flattened = np.concatenate([img_flattened, padding])

                all_X_for_this_output_split.append(img_flattened)
                all_Y_for_this_output_split.append(label)
                images_processed_count += 1
                if images_processed_count > 0 and images_processed_count % 5000 == 0: # Log progress less frequently
                    print(f"    Processed {images_processed_count} images for {output_split_name_label} (from {source_difficulty_pattern})...")

            except Exception as e_entry:
                # print(f"  Error processing entry data {entry_data}: {e_entry}")
                entries_skipped_count +=1
                continue
    
    if entries_skipped_count > 0:
        print(f"  Skipped a total of {entries_skipped_count} entries during processing of {source_difficulty_pattern}.")

    if not all_X_for_this_output_split:
        print(f"WARNING: No images were successfully processed and added for Pathfinder-{resolution} / {source_difficulty_pattern} -> {output_split_name_label}.")
        return

    output_X_path = os.path.join(OUTPUT_PROCESSED_DIR_BASE, f"pathfinder{resolution}_{output_split_name_label}_X.npy")
    output_Y_path = os.path.join(OUTPUT_PROCESSED_DIR_BASE, f"pathfinder{resolution}_{output_split_name_label}_Y.npy")

    final_X_array = np.array(all_X_for_this_output_split, dtype=np.int64)
    final_Y_array = np.array(all_Y_for_this_output_split, dtype=np.int64)

    print(f"  Saving {output_split_name_label} data: X shape {final_X_array.shape}, Y shape {final_Y_array.shape}")
    np.save(output_X_path, final_X_array)
    np.save(output_Y_path, final_Y_array)
    print(f"  Saved to {output_X_path} and {output_Y_path}")


if __name__ == "__main__":
    start_time_total = time.time()
    os.makedirs(OUTPUT_PROCESSED_DIR_BASE, exist_ok=True)

    if not os.path.isdir(LRA_RELEASE_ROOT):
        print(f"ERROR: LRA_RELEASE_ROOT directory not found: {LRA_RELEASE_ROOT}")
        print(f"Please ensure this path points to your '{LRA_RELEASE_ROOT}' directory containing pathfinder<RESOLUTION> folders.")
        exit(1)

    for res in PATHFINDER_RESOLUTIONS_TO_PROCESS:
        print(f"--- Starting Preprocessing for Pathfinder Resolution: {res} ---")
        for output_split_name, source_difficulty_folder in OUTPUT_SPLIT_MAPPING.items():
            source_path_check = os.path.join(LRA_RELEASE_ROOT, f"pathfinder{res}", source_difficulty_folder)
            if os.path.isdir(source_path_check):
                process_single_pathfinder_config(res, source_difficulty_folder, output_split_name)
            else:
                print(f"Warning: Source difficulty folder '{source_difficulty_folder}' for res {res} at '{source_path_check}' not found. Skipping this split generation.")

    end_time_total = time.time()
    print(f"\nTotal preprocessing time: {(end_time_total - start_time_total)/60:.2f} minutes.")
    print(f"Preprocessed .npy files should be in: {OUTPUT_PROCESSED_DIR_BASE}")