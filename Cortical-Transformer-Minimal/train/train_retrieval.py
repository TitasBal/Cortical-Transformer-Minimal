# preprocess_aan.py
import os
import numpy as np
import traceback
import time

# Attempt to import TFDS for ByteTextEncoder
try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
    # Prevent TensorFlow from grabbing all GPU memory, allowing PyTorch to use it.
    tf.config.experimental.set_visible_devices([], 'GPU')
    TFDS_AVAILABLE = True
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow Datasets version: {tfds.__version__}")
except ImportError:
    print("CRITICAL ERROR: tensorflow-datasets or tensorflow is not installed. Please install them.")
    TFDS_AVAILABLE = False
    exit(1)


# --- Configuration ---
# This is the target sequence length for the combined (paper1 + SEP + paper2) byte sequence.
# Adjust based on LRA task specs or your model's capabilities.
# LRA retrieval tasks often deal with very long effective sequences.
# If individual papers are ~4k, two papers + SEP could be ~8k.
# Your model's internal block_size for chunking will be smaller.
LRA_RETRIEVAL_SEQ_LEN: int = 4096 # Example: You might need 8192 or more
                                 # This MUST match what your model expects after chunking.

# For byte-level encoding, the vocab size is 256 (0-255).
# Padding is usually done with 0 for byte sequences if 0 is not a common data byte.
PAD_ID: int = 0

# --- PATHS TO CONFIGURE ---
# **IMPORTANT**: Update these paths to match your system!
# Path to your extracted ORIGINAL AAN dataset (where you find individual paper files)
AAN_SOURCE_DATA_ROOT: str = "D:/lra_datasets/aan_source/acl-arc-160301-json/" # EXAMPLE: Adjust this!
# Path to the folder containing LRA's ID split files (e.g., new_aan_pairs.train.tsv)
LRA_SPLITS_DIR: str = "D:/Downloads/Šveicarija/Minimal-Cortical-Transformer/tsv_data/" # From your screenshot
# Path where the processed .npy files will be saved
OUTPUT_PROCESSED_DIR: str = "D:/lra_datasets/aan_processed_for_pytorch/"
# --- END PATHS ---

# Initialize the ByteTextEncoder
# This encoder takes a Python string and returns a list of integers (0-255)
try:
    BYTE_ENCODER = tfds.deprecated.text.ByteTextEncoder() # Removed reserved_tokens
except Exception as e:
    print(f"Error initializing ByteTextEncoder: {e}")
    print("Ensure tensorflow_datasets is installed and up-to-date.")
    exit(1)

def get_paper_text(paper_id: str, aan_source_root: str) -> str:
    """
    Placeholder function to retrieve the text (e.g., abstract) of a paper
    given its ID from the original AAN dataset.

    YOU MUST IMPLEMENT THIS FUNCTION BASED ON HOW THE AAN DATA IS STRUCTURED.

    Args:
        paper_id: The ID of the paper (e.g., "W08-0101").
        aan_source_root: The root directory of the original AAN dataset.

    Returns:
        The text content of the paper, or an empty string if not found/error.
    """
    # Example: If papers are in JSON files like <paper_id>.json
    # and the text is under a key like "abstract" or "full_text".
    # This is a GUESS - you need to inspect your AAN data.
    # The AAN data from aan.how is often a collection of JSON files per paper,
    # but also a large metadata JSON. You might need to parse the large metadata file first
    # to find paths or directly access text if it's embedded.

    # Let's assume for this example, there's a main JSON file or individual files.
    # This is a VERY simplified placeholder.
    # You'll likely need to:
    # 1. Determine the file naming convention (e.g., P99-1001.json, W08-2202.xml etc.)
    # 2. Determine the sub-directory structure within aan_source_root.
    # 3. Use libraries like `json` or `xml.etree.ElementTree` to parse the files.
    # 4. Extract the relevant text field (e.g., "abstractText", "bodyText", "string_content").

    # --- START OF MODIFIABLE SECTION for get_paper_text ---
    # Try finding a common pattern, e.g. if JSON files are named by ID directly.
    # This path is a GUESS. The AAN dataset has various sub-collections.
    # You might need to search within different subfolders of AAN_SOURCE_DATA_ROOT
    # or parse a central metadata file.
    # The AAN data from `http://aan.how/download/` typically gives a large metadata.json
    # and a directory of individual paper JSONs.
    # Let's assume we are trying to read from individual JSON files.
    # Example: paper_id might be "W08-0101". File might be "W08-0101.json".
    # The structure inside the JSON also matters.
    # A common key for abstract is 'abstractText' or 'abstract'.

    potential_file_path = os.path.join(aan_source_root, f"{paper_id}.json") # GUESSING .json extension
    if not os.path.exists(potential_file_path) and '-' in paper_id:
        # Sometimes IDs in splits might be different from filenames (e.g. with version numbers)
        # Try to construct path like: data/json/W/W08/W08-0101.json
        # This is based on typical AAN ARC structure.
        collection_id = paper_id.split('-')[0][:-2] # e.g., W from W08
        volume_id = paper_id.split('-')[0] # e.g., W08 from W08-0101
        file_path_alt = os.path.join(aan_source_root, collection_id, volume_id, f"{paper_id}.json")
        if os.path.exists(file_path_alt):
            potential_file_path = file_path_alt
        else:
            # Fallback for flat structures sometimes found in simplified AAN versions
            file_path_flat = os.path.join(aan_source_root, "papers_json", f"{paper_id}.json") # Another guess
            if os.path.exists(file_path_flat):
                potential_file_path = file_path_flat


    if os.path.exists(potential_file_path):
        try:
            with open(potential_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Try common keys for abstract or full text
            if 'abstractText' in data and data['abstractText']:
                return str(data['abstractText'])
            elif 'abstract' in data and isinstance(data['abstract'], str) and data['abstract']:
                return data['abstract']
            elif 'fullText' in data and data['fullText']: # Less common for LRA retrieval
                return str(data['fullText'])
            elif 'string_content' in data and data['string_content']: # Another possible key
                 return str(data['string_content'])
            else:
                # print(f"Warning: Paper {paper_id} found but no clear abstract/text field. Keys: {list(data.keys())}")
                return "" # Return empty string if no suitable text found
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON for paper {paper_id} at {potential_file_path}")
            return ""
        except Exception as e:
            print(f"Warning: Error reading paper {paper_id} from {potential_file_path}: {e}")
            return ""
    else:
        # print(f"Warning: Paper file not found for ID {paper_id} (tried {potential_file_path})")
        return ""
    # --- END OF MODIFIABLE SECTION for get_paper_text ---

def process_split(split_file_path: str, aan_source_root_path: str, output_prefix: str):
    """
    Processes an LRA ID split file, fetches paper texts, encodes them,
    and saves them as .npy files.
    """
    print(f"\nProcessing split file: {split_file_path}...")
    processed_X = []
    processed_Y = []
    skipped_pairs = 0

    # Special token to separate the two documents
    # Ensure this token is handled by your model's vocabulary if it's not byte-encoded.
    # For byte encoding, it will just become a sequence of bytes.
    separator_token_text = " [SEP] "

    with open(split_file_path, 'r', encoding='utf-8') as f_split:
        lines = f_split.readlines()

    for line_num, line in enumerate(lines):
        if line_num == 0 and "paper1_id" in line.lower().replace("_", ""): # Try to catch header
            print(f"Skipping header: {line.strip()}")
            continue

        parts = line.strip().split('\t') # Assuming TSV: label paper1_id paper2_id
        if len(parts) != 3:
            # Try splitting by space if tab split failed (some LRA files might use spaces)
            parts = line.strip().split()
            if len(parts) != 3:
                print(f"Warning: Malformed line {line_num + 1} in split file: '{line.strip()}'. Skipping.")
                skipped_pairs +=1
                continue

        try:
            label, p1_id, p2_id = int(parts[0]), parts[1], parts[2]
        except ValueError:
            print(f"Warning: Could not parse label or IDs on line {line_num + 1}: '{line.strip()}'. Skipping.")
            skipped_pairs +=1
            continue

        text1 = get_paper_text(p1_id, aan_source_root_path)
        text2 = get_paper_text(p2_id, aan_source_root_path)

        if not text1 or not text2:
            # print(f"Warning: Missing text for pair ({p1_id}, {p2_id}) on line {line_num + 1}. Skipping pair.")
            skipped_pairs +=1
            continue

        combined_text = text1 + separator_token_text + text2

        # Encode to byte integers using TFDS's ByteTextEncoder
        try:
            encoded_bytes_list = BYTE_ENCODER.encode(combined_text)
            encoded_bytes = np.array(encoded_bytes_list, dtype=np.int64)
        except Exception as e_enc:
            print(f"Warning: Error byte-encoding text for pair ({p1_id}, {p2_id}): {e_enc}. Skipping pair.")
            skipped_pairs +=1
            continue


        # Pad or Truncate the byte sequence
        current_len = len(encoded_bytes)
        if current_len > LRA_RETRIEVAL_SEQ_LEN:
            encoded_bytes = encoded_bytes[:LRA_RETRIEVAL_SEQ_LEN]
        elif current_len < LRA_RETRIEVAL_SEQ_LEN:
            padding = np.full(LRA_RETRIEVAL_SEQ_LEN - current_len, PAD_ID, dtype=np.int64)
            encoded_bytes = np.concatenate([encoded_bytes, padding])

        processed_X.append(encoded_bytes)
        processed_Y.append(label)

        if (line_num + 1) % 1000 == 0: # Log progress
            print(f"  Processed {line_num + 1}/{len(lines)} pairs for {os.path.basename(split_file_path)}...")

    if skipped_pairs > 0:
        print(f"Warning: Skipped {skipped_pairs} pairs due to missing text or parsing errors for {os.path.basename(split_file_path)}.")

    if not processed_X:
        print(f"ERROR: No data was successfully processed for {split_file_path}. Check paths and data.")
        return

    print(f"Converting processed data to NumPy arrays for {os.path.basename(split_file_path)}...")
    final_X = np.array(processed_X, dtype=np.int64)
    final_Y = np.array(processed_Y, dtype=np.int64)

    print(f"Saving {output_prefix}_X.npy (shape: {final_X.shape}) and {output_prefix}_Y.npy (shape: {final_Y.shape})")
    np.save(f"{output_prefix}_X.npy", final_X)
    np.save(f"{output_prefix}_Y.npy", final_Y)
    print(f"Saved {os.path.basename(split_file_path)} data.")

if __name__ == "__main__":
    start_time_total = time.time()
    os.makedirs(OUTPUT_PROCESSED_DIR, exist_ok=True)

    print(f"Using AAN source data from: {AAN_SOURCE_DATA_ROOT}")
    if not os.path.isdir(AAN_SOURCE_DATA_ROOT):
        print(f"ERROR: AAN_SOURCE_DATA_ROOT directory not found: {AAN_SOURCE_DATA_ROOT}")
        print("Please download the original AAN dataset and update the path.")
        exit(1)

    print(f"Using LRA ID splits from: {LRA_SPLITS_DIR}")
    if not os.path.isdir(LRA_SPLITS_DIR):
        print(f"ERROR: LRA_SPLITS_DIR directory not found: {LRA_SPLITS_DIR}")
        print("Please ensure your new_aan_pairs.*.tsv files are in this directory.")
        exit(1)

    print(f"Saving processed .npy files to: {OUTPUT_PROCESSED_DIR}")

    # Define the splits to process
    # Adjust filenames if yours are different (e.g., new_aan_pairs.dev.tsv)
    splits_to_process = {
        "train": "new_aan_pairs.train.tsv",
        "val": "new_aan_pairs.eval.tsv", # LRA often uses 'dev', check your filename
        "test": "new_aan_pairs.test.tsv"
    }

    for split_name, tsv_filename in splits_to_process.items():
        split_file_full_path = os.path.join(LRA_SPLITS_DIR, tsv_filename)
        output_file_prefix = os.path.join(OUTPUT_PROCESSED_DIR, split_name) # e.g., .../train
        if os.path.exists(split_file_full_path):
            process_split(split_file_full_path, AAN_SOURCE_DATA_ROOT, output_file_prefix)
        else:
            print(f"Warning: Split file not found, skipping: {split_file_full_path}")

    end_time_total = time.time()
    print(f"\nTotal preprocessing time: {(end_time_total - start_time_total)/60:.2f} minutes.")
    print(f"Preprocessed .npy files saved in: {OUTPUT_PROCESSED_DIR}")