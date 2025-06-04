# evaluate_lra.py
import argparse
import os
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
import traceback # For printing error details
from torch.utils.data import Dataset, DataLoader # For NpyPathfinderDataset

# Attempt to import TFDS, handle gracefully if not installed
try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
    # Prevent TensorFlow from grabbing all GPU memory
    tf.config.experimental.set_visible_devices([], 'GPU')
    TFDS_AVAILABLE = True
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow Datasets version: {tfds.__version__}")
except ImportError:
    print("Warning: tensorflow_datasets or tensorflow not found. TSV & NPY loading will still work.")
    TFDS_AVAILABLE = False

# Import your model definition
try:
    from model1 import GPTConfig, GPT # Ensure model1.py has the latest GPT
    print("Imported GPTConfig and GPT successfully from model1.py.")
except ImportError as e:
    print(f"Error: Failed to import model definitions from model1.py: {e}")
    exit(1)

# Suppress PyTorch Compile errors if Triton is missing
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# --- ListOps Specific Vocabulary ---
LISTOPS_VOCAB = {
    '<pad>': 0, '[': 1, ']': 2, 'MIN': 3, 'MAX': 4, 'MED': 5, 'SM': 6,
    '0': 7, '1': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15, '9': 16
}
LISTOPS_VOCAB_SIZE = len(LISTOPS_VOCAB)
# ----------------------------------

def encode_listops_string(s, stoi_map):
    tokens = s.split(' ')
    encoded = [stoi_map.get(token, stoi_map['<pad>']) for token in tokens]
    return np.array(encoded, dtype=np.int64)

def load_listops_tsv(filepath):
    print(f"Loading ListOps data from: {filepath}")
    examples = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f: lines = f.readlines()
        for i, line in enumerate(lines[1:]): # Skip header
            line_num = i + 2; parts = line.strip().split('\t')
            if len(parts) == 2:
                input_text, target_label_str = parts[0], parts[1]
                try:
                    target_label = int(target_label_str)
                    if 0 <= target_label <= 9: examples.append({'input_text': input_text, 'target_label': target_label})
                except ValueError: pass
        print(f"Read {len(examples)} valid examples from {os.path.basename(filepath)}.")
        return examples
    except FileNotFoundError: print(f"ERROR: Data file not found at {filepath}"); return None
    except Exception as e: print(f"ERROR: Failed to load TSV from {filepath}: {e}"); traceback.print_exc(); return None

class NpyPathfinderDataset(Dataset): # Renamed for clarity
    def __init__(self, resolution: int, split_name: str, base_dir: str):
        self.x_path = os.path.join(base_dir, f"pathfinder{resolution}_{split_name}_X.npy")
        self.y_path = os.path.join(base_dir, f"pathfinder{resolution}_{split_name}_Y.npy")
        print(f"Attempting to load X data for Pathfinder {split_name} from: {self.x_path}")
        self.x_data = np.load(self.x_path, mmap_mode='r')
        print(f"Attempting to load Y data for Pathfinder {split_name} from: {self.y_path}")
        self.y_data = np.load(self.y_path, mmap_mode='r')
        assert len(self.x_data) == len(self.y_data), "X and Y .npy files must have same number of samples."
        print(f"Loaded {len(self.y_data)} samples for Pathfinder-{resolution} {split_name} split.")
    def __len__(self): return len(self.y_data)
    def __getitem__(self, idx):
        sample_x = torch.from_numpy(self.x_data[idx].astype(np.int64))
        sample_y = torch.tensor(self.y_data[idx].astype(np.int64), dtype=torch.long)
        return {'input': sample_x, 'target': sample_y}

def get_lra_task_config(task_name, resolution=32):
    if task_name == 'listops':
        return {'sequence_length': 2000, 'vocab_size': LISTOPS_VOCAB_SIZE, 'num_classes': 10,
                'task_type': 'classification', 'metric': 'accuracy', 'data_format': 'tsv'}
    elif task_name == 'text':
        return {'sequence_length': 4000, 'vocab_size': 256, 'num_classes': 2,
                'task_type': 'classification', 'metric': 'accuracy',
                'data_format': 'tfds', 'tfds_name': 'imdb_reviews'}
    elif task_name == 'image':
        return {'sequence_length': 1024 * 3, 'vocab_size': 256, 'num_classes': 10,
                'task_type': 'classification', 'metric': 'accuracy',
                'data_format': 'tfds', 'tfds_name': 'cifar10'}
    elif task_name == 'pathfinder':
        return {'sequence_length': resolution * resolution, 'vocab_size': 256, 'num_classes': 2,
                'task_type': 'classification', 'metric': 'accuracy',
                'data_format': 'npy_manual', 'resolution': resolution}
    elif task_name == 'retrieval':
         return {'sequence_length': 8000, 'vocab_size': 256, 'num_classes': 2,
                 'task_type': 'classification', 'metric': 'accuracy', 'data_format': 'npy_manual'} # Assuming npy for retrieval too
    else: raise ValueError(f"Unknown LRA task: {task_name}")

def load_checkpoint_and_config(checkpoint_path, device):
    # ... (load_checkpoint_and_config function remains the same as previous full version) ...
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint: state_dict = checkpoint['model']
        else: state_dict = checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print("State dictionary extracted.")
        config_data = checkpoint.get('config', None)
        if isinstance(config_data, GPTConfig): return state_dict, config_data
        elif isinstance(config_data, dict):
            try: # Try to reconstruct GPTConfig if it was saved as dict
                 valid_keys = {k: v for k, v in config_data.items() if k in GPTConfig.__annotations__}
                 return state_dict, GPTConfig(**valid_keys)
            except TypeError as e:
                 print(f"Warning: Could not instantiate GPTConfig from saved dict: {e}. Returning dict.")
                 return state_dict, config_data # Return dict as fallback
        else: print("Warning: No valid config found in checkpoint."); return state_dict, None
    except FileNotFoundError: print(f"ERROR: Checkpoint file not found at {checkpoint_path}"); exit(1)
    except Exception as e: print(f"ERROR: Failed to load checkpoint: {e}"); traceback.print_exc(); exit(1)

def prepare_batch(batch, task_config, device):
    # ... (prepare_batch function remains the same as previous full version) ...
    task_type = task_config['task_type']
    x, y = None, None
    input_key, target_key = 'input', 'target'
    try:
        if isinstance(batch, dict) and 'input' in batch and 'target' in batch:
            x = batch['input'].to(device); y = batch['target'].to(device)
        elif hasattr(batch, 'keys'):
             input_key = 'inputs' if 'inputs' in batch else 'image' if 'image' in batch else 'text' if 'text' in batch else 'input'
             target_key = 'targets' if 'targets' in batch else 'label' if 'label' in batch else 'target'
             if input_key not in batch or target_key not in batch: raise ValueError(f"Missing keys '{input_key}' or '{target_key}'. Keys: {list(batch.keys())}")
             x_np, y_np = batch[input_key], batch[target_key]
             if task_config.get('tfds_name') == 'cifar10' and input_key == 'image': x_np = x_np.reshape(x_np.shape[0], -1)
             x = torch.from_numpy(x_np).to(torch.long).to(device)
             y = torch.from_numpy(y_np).to(torch.long).to(device)
        elif isinstance(batch, tuple) and len(batch) == 2:
            x_np, y_np = batch[0], batch[1]
            if task_config.get('tfds_name') == 'cifar10' and hasattr(x_np,'ndim') and x_np.ndim > 2: x_np = x_np.reshape(x_np.shape[0] if x_np.ndim > 1 else 1, -1)
            x = torch.from_numpy(x_np).to(torch.long).to(device)
            y = torch.from_numpy(y_np).to(torch.long).to(device)
        else: raise TypeError(f"Unexpected batch data format: {type(batch)}")
        if x.dim() == 1: x = x.unsqueeze(0)
        if task_type == 'classification' and y.dim() != 1: y = y.view(-1)
        return x, y
    except Exception as e: print(f"\nERROR in prepare_batch: {e}"); traceback.print_exc(); return None, None

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT models on LRA tasks.")
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--lra_task', type=str, required=True, choices=['listops', 'text', 'image', 'pathfinder', 'retrieval'])
    parser.add_argument('--pathfinder_resolution', type=int, default=32)
    parser.add_argument('--pathfinder_split_name', type=str, default='test', help="e.g., 'test', 'validation', 'hard'") # For .npy files
    parser.add_argument('--processed_data_dir', type=str, default='./lra_pathfinder_preprocessed/', help="Dir with preprocessed .npy files")
    parser.add_argument('--tfds_data_dir', type=str, default=os.path.expanduser('~/tensorflow_datasets'))
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_dynamic_step', action='store_true')
    parser.add_argument('--compile', action='store_true')
    # Model structure overrides (use if checkpoint doesn't have a complete config)
    parser.add_argument('--n_layer',type=int,default=None); parser.add_argument('--n_head',type=int,default=None)
    parser.add_argument('--n_embd',type=int,default=None); parser.add_argument('--block_size',type=int,default=None)
    parser.add_argument('--dropout',type=float,default=None);
    # Dynamic param overrides (if not in checkpoint config, or to test different eval settings)
    parser.add_argument('--T',type=int,default=None); parser.add_argument('--dt',type=float,default=None)
    parser.add_argument('--tau_att',type=float,default=None); parser.add_argument('--tau_v',type=float,default=None)
    # Removed --use_prospective_coding and --force_prospective_off
    return parser.parse_args()

# --- Main Evaluation Function ---
def main_evaluation():
    args = parse_args()
    torch.manual_seed(42); np.random.seed(42) # Fixed seed for eval
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device != 'cpu' and dtype != torch.float32 else nullcontext()
    print(f"Eval Device: {device}, Dtype: {dtype}, Dynamic: {args.use_dynamic_step}, Compile: {args.compile}")

    try: task_config = get_lra_task_config(args.lra_task, resolution=args.pathfinder_resolution)
    except ValueError as e: print(f"ERROR: {e}"); exit(1)
    print(f"LRA Task: {args.lra_task}, Task Config: {task_config}")

    state_dict_from_checkpoint, config_from_checkpoint = load_checkpoint_and_config(args.checkpoint_path, device)

    # Determine final config: Start with defaults, then checkpoint, then CLI overrides
    # Default config values (should match your GPTConfig defaults or sensible fallbacks)
    final_config_args = {
        'block_size': 512, 'vocab_size': task_config['vocab_size'], # vocab_size MUST match task
        'n_layer': 8, 'n_head': 8, 'n_embd': 512,
        'dropout': 0.0, # Default to no dropout for evaluation
        'tau_att': 1.0, 'tau_v': 1.0, 'dt': 0.01, 'T': 1
        # use_prospective_coding is no longer set here, taken from model's GPTConfig default
    }
    if isinstance(config_from_checkpoint, GPTConfig):
        print("Applying config stored in checkpoint (GPTConfig object)...")
        # Update final_config_args with checkpoint values, but CLI args will override later
        for key, val in config_from_checkpoint.__dict__.items():
            if key in final_config_args: final_config_args[key] = val
    elif isinstance(config_from_checkpoint, dict):
        print("Applying config stored in checkpoint (dictionary)...")
        for key, val in config_from_checkpoint.items():
            if key in final_config_args: final_config_args[key] = val
    else:
        print("Warning: No valid config found in checkpoint. Using script defaults overridden by CLI.")

    # Apply command-line overrides
    overridden_params = {}
    for arg_name in ['n_layer','n_head','n_embd','block_size','dropout','T','dt','tau_att','tau_v']:
        cli_val = getattr(args, arg_name)
        if cli_val is not None: final_config_args[arg_name] = cli_val; overridden_params[arg_name] = cli_val
    
    # Create model with the determined config
    model_config_final = GPTConfig(**final_config_args)
    # Your GPT model must handle num_classes correctly for classification
    model = GPT(model_config_final, num_classes=task_config.get('num_classes'))
    print(f"Final model config for evaluation: {model.config}")
    if overridden_params: print(f"Applied overrides from command line: {overridden_params}")

    load_result = model.load_state_dict(state_dict_from_checkpoint, strict=False)
    print(f"Weight load - Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
    model.to(device); model.eval()

    if args.compile:
        if hasattr(torch, 'compile'): print("Compiling model..."); model = torch.compile(model, mode='reduce-overhead'); print("Model compiled.")
        else: print("torch.compile not available.")

    # --- Load LRA Dataset ---
    eval_dataloader, num_batches_approx = None, "?"
    data_format = task_config.get('data_format')

    if data_format == 'tsv': # ListOps
        # ... (TSV loading for ListOps test set as before) ...
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct path to where lra_release (containing listops-1000) is relative to script
        data_root_listops = os.path.join(script_dir, 'lra_release', 'lra_release')
        test_file = os.path.join(data_root_listops, 'listops-1000', 'basic_test.tsv')
        loaded_examples = load_listops_tsv(test_file)
        if loaded_examples is None: exit(1)
        eval_dataset = [{'input': torch.from_numpy(encode_listops_string(ex['input_text'], LISTOPS_VOCAB)),
                         'target': torch.tensor(ex['target_label'], dtype=torch.long)} for ex in loaded_examples]
        def collate_fn_listops(batch):
            inputs = [item['input'] for item in batch]; targets = [item['target'] for item in batch]
            max_len_batch = max(len(inp) for inp in inputs)
            padded_inputs = [F.pad(inp, (0, max_len_batch - len(inp)), value=LISTOPS_VOCAB['<pad>']) for inp in inputs]
            return {'input': torch.stack(padded_inputs), 'target': torch.stack(targets)}
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn_listops)
        num_batches_approx = len(eval_dataloader)
        print(f"Prepared {num_batches_approx} ListOps batches for evaluation.")

    elif data_format == 'npy_manual': # Pathfinder or AAN Retrieval from preprocessed .npy
        split_to_eval = args.pathfinder_split_name if args.lra_task == 'pathfinder' else 'test'
        try:
            dataset_to_eval = NpyPathfinderDataset(task_config['resolution'], split_to_eval, args.processed_data_dir)
            eval_dataloader = DataLoader(dataset_to_eval, batch_size=args.batch_size, shuffle=False, num_workers=0)
            num_batches_approx = len(eval_dataloader)
            print(f"Prepared DataLoader for {args.lra_task} '{split_to_eval}' split with {num_batches_approx} batches.")
        except FileNotFoundError as e: print(f"ERROR: .npy files for {args.lra_task} {split_to_eval}. Missing: {e.filename}"); exit(1)

    elif data_format == 'tfds': # For Text (IMDb), Image (CIFAR)
        # ... (TFDS loading logic as corrected previously, using ByteTextEncoder for text) ...
        if not TFDS_AVAILABLE: print("ERROR: tensorflow_datasets required."); exit(1)
        tfds_base_name = task_config['tfds_name']
        print(f"Loading TFDS: {tfds_base_name} (test split) from {args.tfds_data_dir}")
        try:
            split_to_load = 'test'
            data_processed_list = []
            if tfds_base_name == 'imdb_reviews':
                byte_encoder = tfds.deprecated.text.ByteTextEncoder()
                def python_preprocess_text(text_tensor_np, label_tensor_np):
                    text_str = text_tensor_np.decode('utf-8'); encoded_bytes = np.array(byte_encoder.encode(text_str), dtype=np.int64)
                    current_len = len(encoded_bytes); target_len = task_config['sequence_length']
                    if current_len > target_len: encoded_bytes = encoded_bytes[:target_len]
                    elif current_len < target_len: encoded_bytes = np.concatenate([encoded_bytes, np.full(target_len - current_len, 0, dtype=np.int64)])
                    return {'input': encoded_bytes, 'target': label_tensor_np}
                ds_tf, ds_info = tfds.load(name=tfds_base_name, data_dir=args.tfds_data_dir, as_supervised=True, split=split_to_load, shuffle_files=False, with_info=True)
                for text_np, lab_np in ds_tf.as_numpy_iterator(): data_processed_list.append(python_preprocess_text(text_np, lab_np))
            elif tfds_base_name == 'cifar10':
                def python_preprocess_cifar(image_np, label_np):
                    img_flattened = image_np.reshape(-1).astype(np.int64) # Flatten (H,W,C) to (L,)
                    # Ensure fixed length LRA_SEQ_LEN (3072 for CIFAR10) - usually not needed as images are fixed
                    target_len = task_config['sequence_length']
                    if len(img_flattened) != target_len: # Should not happen for CIFAR10
                        raise ValueError(f"CIFAR10 image flattened to {len(img_flattened)} not {target_len}")
                    return {'input': img_flattened, 'target': label_np}
                ds_tf, ds_info = tfds.load(name=tfds_base_name, data_dir=args.tfds_data_dir, as_supervised=True, split=split_to_load, shuffle_files=False, with_info=True)
                for img_np, lab_np in ds_tf.as_numpy_iterator(): data_processed_list.append(python_preprocess_cifar(img_np, lab_np))
            else: raise ValueError(f"TFDS preprocessing not defined for {tfds_base_name}")

            class TempDictDataset(Dataset): # Simple dataset from list of dicts
                def __init__(self, data_list): self.data_list = data_list
                def __len__(self): return len(self.data_list)
                def __getitem__(self, idx): return self.data_list[idx]
            eval_dataset = TempDictDataset(data_processed_list)
            eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
            num_test_samples = ds_info.splits[split_to_load].num_examples
            num_batches_approx = math.ceil(num_test_samples / args.batch_size)
            print(f"TFDS '{tfds_base_name}' ({split_to_load}) processed into {num_batches_approx} batches.")
        except Exception as e: print(f"ERROR loading TFDS '{tfds_base_name}': {e}"); traceback.print_exc(); exit(1)
    else: print(f"ERROR: Unknown data_format: {task_config.get('data_format')}"); exit(1)

    if eval_dataloader is None: print("ERROR: Eval DataLoader not created."); exit(1)

    # --- Evaluation Loop ---
    total_samples, correct_predictions, batch_times = 0, 0, []
    torch.cuda.reset_peak_memory_stats(device); peak_memory = 0
    print(f"Starting evaluation on approx. {num_batches_approx} batches...")
    forward_method = model.step if args.use_dynamic_step else model.forward

    with torch.no_grad():
        for batch_num, batch_data in enumerate(eval_dataloader): # Use DataLoader
            t_batch_start = time.time()
            # prepare_batch now directly takes the dict from DataLoader
            X_full, Y_true = prepare_batch(batch_data, task_config, device)
            if X_full is None: print(f"Warning: Skipping batch {batch_num+1} data prep error."); continue

            B, T_seq_full = X_full.shape; total_samples += B
            model_block_size = model.config.block_size
            num_chunks = math.ceil(T_seq_full / model_block_size)
            last_token_logits = None
            for i in range(num_chunks):
                start_idx, end_idx = i*model_block_size, min((i+1)*model_block_size, X_full.size(1))
                X_chunk = X_full[:, start_idx:end_idx]
                with ctx:
                    logits_from_model_for_chunk, _ = forward_method(X_chunk, targets=None)
                if task_config['task_type'] == 'classification' and i == num_chunks - 1:
                    last_token_logits = logits_from_model_for_chunk
            if task_config['task_type'] == 'classification':
                if last_token_logits is None: print(f"Warning: No logits for batch {batch_num+1}."); continue
                predictions = torch.argmax(last_token_logits, dim=-1)
                correct_predictions += (predictions == Y_true.view(-1)).sum().item()
            t_batch_end = time.time(); batch_times.append(t_batch_end - t_batch_start)
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
            if (batch_num + 1) % 50 == 0: print(f"  Processed batch {batch_num+1}/{num_batches_approx}...")

    # --- Calculate Final Metrics ---
    print("\nEvaluation Finished.")
    metric_name = task_config['metric']
    if metric_name == 'accuracy':
        final_metric_val = (correct_predictions / total_samples * 100.0) if total_samples > 0 else 0.0
        print(f"Accuracy: {final_metric_val:.2f}%")
    else: print(f"Metric '{metric_name}' calculation not implemented.")
    total_eval_time = sum(batch_times); avg_time_per_batch = np.mean(batch_times) if batch_times else 0
    throughput_samples = total_samples / total_eval_time if total_eval_time > 0 else 0
    print(f"Total evaluation time: {total_eval_time:.2f}s"); print(f"Average time per batch: {avg_time_per_batch:.3f}s")
    print(f"Throughput: {throughput_samples:.2f} samples/sec"); print(f"Peak GPU Memory: {peak_memory / 1e9:.3f} GB")

if __name__ == "__main__":
    main_evaluation()