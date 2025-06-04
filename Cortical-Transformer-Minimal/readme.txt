-----------------------------------------------------------------------
Instructions for Running LRA Tasks with Your PyTorch GPT Model
-----------------------------------------------------------------------

**I. General Setup (One Time):**

1.  **Environment:** Ensure you have a Python virtual environment with:
    *   PyTorch (with CUDA if using GPU)
    *   NumPy
    *   Matplotlib (for plotting in training scripts)
    *   Pillow (for image processing in preprocess_pathfinder.py)
    *   TensorFlow & TensorFlow Datasets (`pip install tensorflow tensorflow-datasets`)
    *   (Optional) Wandb (`pip install wandb` and `wandb login`)

2.  **Model File:** Your main model code (GPTConfig, GPT class, etc.) should be in `model1.py`.
    *   Ensure `GPT.__init__` in `model1.py` accepts `num_classes: Optional[int] = None`.
    *   Ensure `GPT._get_logits_and_loss` handles both classification and language modeling.
    *   Decide if dynamic mode in `model1.py` *always* uses prospective coding or not. The training scripts assume it's a fixed behavior for `dynamic=True`.

3.  **LRA Data:**
    *   **LRA Release Folder:** You need the `lra_release` folder containing subdirectories like `listops-1000`, `pathfinder32`, etc. (with their `metadata` and `imgs` folders).
        *   Assumed path (adjust in scripts): `./lra_release/lra_release/lra_release/`
    *   **TFDS Cache:** TensorFlow Datasets will download data (like IMDb) to a cache directory.
        *   Default: `~/tensorflow_datasets/` (configurable in scripts via `data_dir`).

**II. Task-Specific Preparations:**

**A. ListOps Task:**
    *   No special preprocessing needed beyond what `train_listops.py` does (reading TSV).
    *   Ensure `lra_data_root` in `train_listops.py` points to your `listops-1000` TSV files.

**B. Pathfinder Task (e.g., Pathfinder32):**
    1.  **Run Preprocessing Script (ONCE per resolution/split config):**
        *   Script: `preprocess_pathfinder.py`
        *   **Configure `preprocess_pathfinder.py`:**
            *   Set `PATHFINDER_RESOLUTIONS_TO_PROCESS = [32]` (or other resolutions).
            *   Set `LRA_RELEASE_ROOT` correctly (e.g., `'./lra_release/lra_release/lra_release/'`).
            *   Set `OUTPUT_PROCESSED_DIR_BASE` (e.g., `'./lra_pathfinder_preprocessed/'`).
            *   Verify/Adjust `OUTPUT_SPLIT_MAPPING` to define how LRA difficulties map to your `train`, `validation`, `test` .npy files.
            *   **CRITICAL:** Inspect your `metadata/*.npy` files and adjust the parsing logic in `load_and_parse_lra_metadata_file` if needed.
        *   Execute: `python preprocess_pathfinder.py`
        *   This will create files like `pathfinder32_train_X.npy`, `pathfinder32_train_Y.npy`, etc., in `OUTPUT_PROCESSED_DIR_BASE`.

**C. Text Classification Task (IMDb Bytes):**
    *   TFDS will handle downloading.
    *   Ensure `data_dir` in `train_text.py` points to your TFDS cache.
    *   The script uses `tfds.deprecated.text.ByteTextEncoder` for byte-level processing.

**III. Training Models:**

*   **General:** Edit the configuration section at the top of the respective training script (`train_listops.py`, `train_text.py`, `train_path.py`) to set model size, dynamic mode, learning rate, batch size, accumulation steps, max iterations, etc.

*   **Command Examples (run from `Minimal-Cortical-Transformer` directory):**

    1.  **Train Static Model on ListOps:**
        *   Edit `train_listops.py`: Set `dynamic: bool = False`, adjust other params.
        *   Run: `python train_listops.py`

    2.  **Train Dynamic Model on ListOps:**
        *   Edit `train_listops.py`: Set `dynamic: bool = True`, configure `tau_att`, `tau_v`, `dt`, `T`.
        *   Run: `python train_listops.py`

    3.  **Train Static Model on LRA Text (IMDb Bytes):**
        *   Edit `train_text.py`: Set `dynamic: bool = False`, adjust params (LR, batch size, block_size for chunking, etc.).
        *   Run: `python train_text.py`

    4.  **Train Dynamic Model on LRA Text (IMDb Bytes):**
        *   Edit `train_text.py`: Set `dynamic: bool = True`, adjust dynamic params.
        *   Run: `python train_text.py`

    5.  **Train Static Model on Pathfinder32 (using preprocessed .npy files):**
        *   Ensure `preprocess_pathfinder.py` has been run for resolution 32.
        *   Edit `train_path.py`:
            *   Set `dynamic: bool = False`.
            *   Set `PATHFINDER_RESOLUTION: int = 32`.
            *   Ensure `PROCESSED_DATA_DIR` points to where the `.npy` files are.
            *   Adjust other params (LR will need to be very low, `block_size` for chunking).
        *   Run: `python train_path.py`

    6.  **Train Dynamic Model on Pathfinder32 (using preprocessed .npy files):**
        *   Edit `train_path.py`: Set `dynamic: bool = True`, dynamic params.
        *   Run: `python train_path.py`

*   **Output:** Trained models (`best_model.pt`, `final_model.pt`) and plots will be saved in subdirectories within `output_dir_base` defined in each script.

**IV. Evaluating Models (on LRA Test Sets):**

*   **Script:** `evaluate_lra.py`.
*   This script loads a pre-trained checkpoint and evaluates it on an LRA task's *test* split.

*   **Command Examples:**

    1.  **Evaluate ListOps Model:**
        *   Ensure `get_lra_task_config` and data loading for `listops` (TSV) are correct in `evaluate_lra.py`.
        *   Run:
            ```bash
            python evaluate_lra.py --checkpoint_path ./listops_output/LISTOPS_RUN_NAME/best_model.pt --lra_task listops [--use_dynamic_step if dynamic model]
            ```

    2.  **Evaluate Text (IMDb) Model:**
        *   Ensure `get_lra_task_config` and TFDS loading for `text` (IMDb, byte-processed) are correct in `evaluate_lra.py`.
        *   Run:
            ```bash
            python evaluate_lra.py --checkpoint_path ./lra_text_output/LRATEXT_RUN_NAME/best_model.pt --lra_task text [--use_dynamic_step if dynamic model] --tfds_data_dir path/to/your/tfds_cache
            ```

    3.  **Evaluate Pathfinder32 Model (using preprocessed .npy files):**
        *   Ensure `get_lra_task_config` points to `data_format: 'npy_manual'`.
        *   Ensure `NpyPathfinderDataset` class is in `evaluate_lra.py`.
        *   Ensure the script loads the `test` split `.npy` files (e.g., `pathfinder32_test_X.npy`). You might need to adjust the `split_name` argument for `NpyPathfinderDataset` or add a specific `--eval_split_name` CLI argument. The current script has `--pathfinder_split_name` which defaults to 'test'.
        *   Run:
            ```bash
            python evaluate_lra.py --checkpoint_path ./lra_pathfinder32_output_from_npy/LRAPATHFINDER_RUN_NAME/best_model.pt --lra_task pathfinder --pathfinder_resolution 32 --processed_data_dir ./lra_pathfinder_preprocessed/ [--use_dynamic_step if dynamic model]
            ```

**General Tips:**

*   **Start Small:** For new tasks like Pathfinder, start with the smallest resolution (e.g., 32), smallest model size, and few iterations to debug the data pipeline and basic training.
*   **Monitor GPU Usage:** Use `nvidia-smi` to check memory usage.
*   **Iterate:** Expect to tune hyperparameters (LR, batch size, block size, accumulation steps, model size, dynamic params) for each LRA task.
*   **Consistency:** Ensure vocabulary, sequence lengths, and preprocessing are handled consistently between your training and evaluation setups for a given task.