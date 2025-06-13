# train_lra_text.py

from contextlib import nullcontext
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
import time
import os
import traceback
import random
import math

# Attempt to import TFDS
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

""" Training script for a GPT model on LRA Text Classification (IMDb Bytes).
    Assumes the dynamic model always uses prospective coding if dynamic=True.
"""

try:
    # Change 'model_dynamic' to 'model1' if that's your actual filename
    from model1 import GPTConfig, GPT # Ensure this imports your model with _breve always on for dynamic
    print("Imported GPTConfig and GPT successfully.")
except ImportError:
    print("Error: Failed to import GPTConfig and GPT from model_dynamic.py (or model1.py).")
    print("Ensure the model definition file is in the same directory or your PYTHONPATH.")
    exit(1)

# Suppress PyTorch Compile errors if Triton is missing (optional)
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

# --- Configuration ---
# Model Args
n_layer: int = 8
n_head: int = 8
n_embd: int = 512
block_size: int = 512
dropout: float = 0.1

# --- LRA Text Specific Config ---
LRA_TASK_NAME: str = 'imdb_reviews'
LRA_SEQ_LEN: int = 4000
VOCAB_SIZE: int = 256
NUM_CLASSES: int = 2

# Dynamic Args
dynamic: bool = True      # <<< Set to True to use model.step() (which includes prospective coding)
# Parameters for the dynamic mode (which uses prospective coding)
# These are inspired by the paper if dynamic is True.
tau_att: float = 2.0     # Paper-like value if using dynamics
tau_v: float = 2.0       # Paper-like value
dt: float = 0.1           # Paper-like value
T: int = 1                # Start with T=1 for speed, paper suggests T=10

# Training Args
batch_size: int = 2
accumulation_steps: int = 8
lr: float = 1e-4
weight_decay: float = 0.1
max_iters: int = 10000
eval_interval: int = 250
eval_batches: int = 50
log_interval: int = 20

# Other
seed: int = 1337
data_dir: str = os.path.expanduser('~/tensorflow_datasets')
output_dir_base: str = './lra_text_output_prospective' # Indicate prospective coding in output
# --- End Configuration ---

def main():
    """Runs the main training loop for LRA Text Classification."""

    # --- Dynamic Run Name ---
    # If dynamic, it implies prospective coding is used (as per model assumption)
    dynamics_label = f"Dynamic_Prospective(T={T},dt={dt},tau={tau_att})" if dynamic else "Static"
    effective_bs = batch_size * accumulation_steps
    run_name = f"LRAText_{dynamics_label}_L{n_layer}_H{n_head}_E{n_embd}_B{effective_bs}_lr{lr}"
    output_dir = os.path.join(output_dir_base, run_name)
    print(f"Run Name: {run_name}")
    print(f"Output Dir: {output_dir}")

    # --- Setup ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device != 'cpu' else nullcontext()
    print(f"Device: {device}, Dtype: {dtype}, Dynamic Mode: {dynamic}")
    if dynamic:
        print(f"Dynamic mode implies prospective coding is active in the model.")
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading (Using TFDS) ---
    print(f"Loading {LRA_TASK_NAME} (byte-processed) from TFDS (data_dir: {data_dir})...")
    try:
        train_split_str = 'train[:80%]'
        val_split_str = 'train[80%:]'

        byte_encoder = tfds.deprecated.text.ByteTextEncoder() # Removed reserved_tokens

        def python_preprocess_and_pad(text_tensor_np, label_tensor_np):
            text_str = text_tensor_np.decode('utf-8')
            encoded_bytes = np.array(byte_encoder.encode(text_str), dtype=np.int64)
            current_len = len(encoded_bytes)
            if current_len > LRA_SEQ_LEN:
                encoded_bytes = encoded_bytes[:LRA_SEQ_LEN]
            elif current_len < LRA_SEQ_LEN:
                padding_value = 0
                padding = np.full(LRA_SEQ_LEN - current_len, padding_value, dtype=np.int64)
                encoded_bytes = np.concatenate([encoded_bytes, padding])
            return {'input': encoded_bytes, 'target': label_tensor_np}

        common_load_args = {
            'name': LRA_TASK_NAME,
            'data_dir': data_dir,
            'as_supervised': True
        }
        print("Loading and preprocessing train data (Python loop)...")
        train_ds_tf = tfds.load(**common_load_args, split=train_split_str, shuffle_files=True)
        train_data_processed = [python_preprocess_and_pad(text_np, lab_np) for text_np, lab_np in train_ds_tf.as_numpy_iterator()]
        print("Loading and preprocessing validation data (Python loop)...")
        val_ds_tf = tfds.load(**common_load_args, split=val_split_str, shuffle_files=False)
        val_data_processed = [python_preprocess_and_pad(text_np, lab_np) for text_np, lab_np in val_ds_tf.as_numpy_iterator()]
        print(f"Loaded and processed {len(train_data_processed)} training examples, {len(val_data_processed)} validation examples.")
    except Exception as e:
        print(f"ERROR: Failed to load or preprocess data from TFDS: {e}"); traceback.print_exc(); return

    # --- Model Initialization ---
    # The use_prospective_coding flag is removed from model_conf_args as it's assumed
    # the model handles this based on the 'dynamic' flag or its internal logic
    model_conf_args = dict(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        block_size=block_size, vocab_size=VOCAB_SIZE, dropout=dropout,
        tau_att=tau_att, tau_v=tau_v, dt=dt, T=T
    )
    config = GPTConfig(**model_conf_args)
    model = GPT(config, num_classes=NUM_CLASSES)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # --- Optimizer & Scheduler ---
    try:
        optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=lr, betas=(0.9, 0.95), device_type=device)
    except AttributeError:
        print("Warning: model.configure_optimizers not found. Creating AdamW manually.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=lr * 0.1)
    grad_clip = 1.0

    # --- Batching Function (same as before) ---
    data_indices = {'train': list(range(len(train_data_processed))), 'val': list(range(len(val_data_processed)))}
    def get_pytorch_batch(split: str = 'train') -> tuple[torch.Tensor, torch.Tensor]:
        data_source = train_data_processed if split == 'train' else val_data_processed
        current_indices_pool = data_indices[split]
        if len(current_indices_pool) < batch_size:
            raise ValueError(f"Dataset split '{split}' has {len(current_indices_pool)} samples remaining, less than batch_size {batch_size}")
        sampled_indices = random.sample(current_indices_pool, batch_size)
        batch_examples = [data_source[i] for i in sampled_indices]
        x_list = [ex['input'] for ex in batch_examples]
        y_list = [ex['target'] for ex in batch_examples]
        x = torch.tensor(np.stack(x_list), dtype=torch.long).to(device)
        y = torch.tensor(np.array(y_list), dtype=torch.long).to(device)
        return x, y

    # --- Loss Estimation Function (same as before) ---
    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        out = {}
        model.eval()
        for split_name in ['train_eval', 'val']:
            data_iter_source = train_data_processed if split_name == 'train_eval' else val_data_processed
            temp_indices = list(range(len(data_iter_source)))
            if not temp_indices:
                 out[f'{split_name}_loss'] = float('nan'); out[f'{split_name}_acc'] = 0.0; continue
            losses_val = torch.zeros(eval_batches)
            correct_val, total_val = 0, 0
            for k in range(eval_batches):
                try:
                    eval_batch_indices = random.sample(temp_indices, min(batch_size, len(temp_indices)))
                    if not eval_batch_indices: break
                    eval_batch_examples = [data_iter_source[i] for i in eval_batch_indices]
                    x_list_eval = [ex['input'] for ex in eval_batch_examples]
                    y_list_eval = [ex['target'] for ex in eval_batch_examples]
                    X_full, Y_true = torch.tensor(np.stack(x_list_eval), dtype=torch.long).to(device), \
                                     torch.tensor(np.array(y_list_eval), dtype=torch.long).to(device)
                    final_chunk_logits = None
                    num_chunks = math.ceil(X_full.size(1) / block_size)
                    for i in range(num_chunks):
                        start_idx, end_idx = i * block_size, min((i + 1) * block_size, X_full.size(1))
                        X_chunk = X_full[:, start_idx:end_idx]
                        with ctx:
                            logits_chunk, _ = model.step(X_chunk, targets=None) if dynamic else model(X_chunk, targets=None)
                        if i == num_chunks - 1: final_chunk_logits = logits_chunk
                    if final_chunk_logits is not None:
                        chunk_loss = F.cross_entropy(final_chunk_logits.view(-1, NUM_CLASSES), Y_true.view(-1))
                        losses_val[k] = chunk_loss.item()
                        pred = torch.argmax(final_chunk_logits, dim=-1)
                        correct_val += (pred == Y_true).sum().item()
                        total_val += Y_true.size(0)
                    else: losses_val[k] = float('nan')
                except Exception as e_est:
                    print(f"Error during {split_name} estimate_loss batch {k}: {e_est}"); losses_val[k] = float('nan')
            valid_losses_val = losses_val[~torch.isnan(losses_val)]
            out[f'{split_name}_loss'] = valid_losses_val.mean().item() if len(valid_losses_val) > 0 else float('nan')
            out[f'{split_name}_acc'] = (correct_val / total_val * 100.0) if total_val > 0 else 0.0
        model.train()
        return out

    # --- Training Loop (same logic as before) ---
    print(f"\nStarting LRA Text training for {max_iters} optimizer steps...")
    start_time = time.time()
    micro_batch_iter_num, optimizer_steps, best_val_acc = 0, 0, -1.0
    running_train_loss_sum = 0.0
    optimizer.zero_grad(set_to_none=True)
    current_train_indices = list(data_indices['train'])
    random.shuffle(current_train_indices)
    train_idx_pos = 0

    try:
        while optimizer_steps < max_iters:
            if optimizer_steps > 0 and optimizer_steps % eval_interval == 0:
                metrics = estimate_loss()
                val_loss = metrics.get('val_loss', float('inf'))
                val_acc = metrics.get('val_acc', 0.0)
                train_eval_acc = metrics.get('train_eval_acc', 0.0)
                current_lr_log = optimizer.param_groups[0]['lr']
                print(f"Opt Step {optimizer_steps}/{max_iters}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Train Eval Acc={train_eval_acc:.2f}%, LR={current_lr_log:.2e}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    checkpoint = {'model_state_dict': model.state_dict(), 'config': config,
                                  'optimizer_step': optimizer_steps, 'best_val_acc': best_val_acc,
                                  'dynamic': dynamic} # Removed use_prospective_coding
                    ckpt_path = os.path.join(output_dir, f'best_model_step{optimizer_steps}.pt')
                    print(f"Saving best model with val_acc {best_val_acc:.2f}% to {ckpt_path}")
                    torch.save(checkpoint, ckpt_path)

            t0 = time.time()
            batch_for_iter_indices = []
            for _ in range(batch_size):
                if train_idx_pos >= len(current_train_indices):
                    print(f"Opt Step {optimizer_steps}: Epoch finished. Reshuffling training data.")
                    random.shuffle(current_train_indices); train_idx_pos = 0
                batch_for_iter_indices.append(current_train_indices[train_idx_pos]); train_idx_pos += 1
            batch_examples = [train_data_processed[i] for i in batch_for_iter_indices]
            x_list_train = [ex['input'] for ex in batch_examples]
            y_list_train = [ex['target'] for ex in batch_examples]
            X_full, Y_true = torch.tensor(np.stack(x_list_train),dtype=torch.long).to(device), \
                             torch.tensor(np.array(y_list_train),dtype=torch.long).to(device)

            final_chunk_logits_train, loss_for_step = None, None
            num_chunks = math.ceil(X_full.size(1) / block_size)
            for i in range(num_chunks):
                start_idx, end_idx = i * block_size, min((i + 1) * block_size, X_full.size(1))
                X_chunk = X_full[:, start_idx:end_idx]
                with ctx:
                    logits_chunk, _ = model.step(X_chunk, targets=None) if dynamic else model(X_chunk, targets=None)
                if i == num_chunks - 1: final_chunk_logits_train = logits_chunk
            if final_chunk_logits_train is not None:
                loss_for_step = F.cross_entropy(final_chunk_logits_train.view(-1, NUM_CLASSES), Y_true.view(-1))

            if loss_for_step is not None and not torch.isnan(loss_for_step):
                loss_item_unscaled = loss_for_step.item()
                running_train_loss_sum += loss_item_unscaled
                loss_for_step = loss_for_step / accumulation_steps
                loss_for_step.backward()
                if (micro_batch_iter_num + 1) % accumulation_steps == 0:
                    if grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    optimizer_steps += 1
                    if optimizer_steps % log_interval == 0 and optimizer_steps > 0:
                        avg_train_loss = running_train_loss_sum / (log_interval * accumulation_steps)
                        current_lr_log = optimizer.param_groups[0]['lr']
                        print(f"Opt Step {optimizer_steps}: Avg Train Loss = {avg_train_loss:.4f}, LR = {current_lr_log:.2e}")
                        running_train_loss_sum = 0.0
            else: print(f"Warning: Invalid loss for micro_batch_iter {micro_batch_iter_num}. Skipping.")
            t1 = time.time()
            if (micro_batch_iter_num + 1) % (100 * accumulation_steps) == 0 and micro_batch_iter_num > 0:
                 print(f"MicroIter {micro_batch_iter_num} (Opt Step {optimizer_steps}): micro-batch proc. time {(t1-t0)*1000:.2f}ms")
            micro_batch_iter_num += 1
    except KeyboardInterrupt: print("\nTraining interrupted.")
    except Exception as e: print(f"\nError during training: {e}"); traceback.print_exc()
    finally:
        print("\nTraining finished or stopped.")
        elapsed_time = time.time() - start_time
        print(f"Total training time: {elapsed_time:.2f}s for {optimizer_steps} optimizer steps.")
        final_checkpoint = {'model_state_dict': model.state_dict(), 'config': config,
                            'optimizer_step': optimizer_steps, 'best_val_acc': best_val_acc,
                            'dynamic': dynamic} # Removed use_prospective_coding
        ckpt_path = os.path.join(output_dir, 'final_model.pt')
        print(f"Saving final model to {ckpt_path}")
        torch.save(final_checkpoint, ckpt_path)

if __name__ == "__main__":
    main()