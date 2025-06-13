# train_listops.py (renamed for clarity)

from contextlib import nullcontext
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
import traceback
import random # For shuffling dataset indices


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Suppress PyTorch Compile errors if Triton is missing (optional)
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

""" Training script for a GPT model (static or dynamic) on ListOps. """

try:
    # Assuming model_dynamic.py has the necessary GPTConfig, GPT classes
    # AND the GPT class __init__ accepts num_classes and uses a classification head if provided.
    from model1 import GPTConfig, GPT
except ImportError:
    print("Error: Failed to import GPTConfig and GPT from model_dynamic.py.")
    exit(1)

# --- Configuration ---
# Model Args
n_layer: int = 8
n_head: int = 8
n_embd: int = 512
# block_size: How much context the MODEL sees internally per step/chunk.
# ListOps sequences can be long, but we'll likely pad batches dynamically.
# Set this based on expected processing chunk size or max length within a batch if not chunking model internally.
# Let's assume model handles sequences up to this length.
block_size: int = 2048 # Increased context window for potentially longer ListOps sub-sequences
dropout: float = 0.1

# --- ListOps Specific Config ---
LISTOPS_VOCAB = { # Double-check this vocab based on LRA code/data
    '<pad>': 0, '[': 1, ']': 2, 'MIN': 3, 'MAX': 4, 'MED': 5, 'SM': 6, # SUM_MOD
    '0': 7, '1': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15, '9': 16
}
LISTOPS_VOCAB_SIZE = len(LISTOPS_VOCAB)
LISTOPS_NUM_CLASSES = 10 # Output digits 0-9

# Dynamic Args
dynamic: bool = False # <<< Set to True to use model.step()
# Using simplified dynamic params that worked best before
tau_att: float = 2.0
tau_v: float = 2.0
dt: float = 0.1
T: int = 1

# Training Args
batch_size: int = 2      # May need smaller due to longer sequences

accumulation_steps: int = 2

lr: float = 1e-4          # Start with known good LR
weight_decay: float = 0.1

optimizer_steps_target: int = 3000

max_iters: int = optimizer_steps_target * accumulation_steps    # ListOps might need more iterations
eval_interval: int = 100  # Evaluate more often
eval_batches: int = 50    # Use more batches for eval
log_interval: int = 10    # Log train loss more often

# Other
seed: int = 1337
# Assume lra_release folder is accessible from where script is run
lra_data_root: str = './lra_release/lra_release' # Adjust this path if needed
output_dir_base: str = './listops_output'
# --- End Configuration ---

def encode_listops_string(s, stoi_map):
    """Encodes a ListOps string sequence into integer indices."""
    tokens = s.split(' ')
    encoded = [stoi_map.get(token, stoi_map['<pad>']) for token in tokens]
    return np.array(encoded, dtype=np.int64)

def load_listops_tsv(filepath):
    """Loads ListOps data from a TSV file, skipping the header.""" # Updated docstring
    print(f"Loading ListOps data from: {filepath}")
    examples = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # --- ADDED: Skip the header line (the first line) ---
        for i, line in enumerate(lines[1:]): # Start iterating from the second line (index 1)
            line_num = i + 2 # Adjust line number for error messages
            # --------------------------------------------------
            parts = line.strip().split('\t')
            if len(parts) == 2:
                input_text = parts[0]
                try:
                    target_label = int(parts[1])
                    if 0 <= target_label <= 9: # Validate label range
                         examples.append({'input_text': input_text, 'target_label': target_label})
                    else:
                         print(f"Warning: Skipping line {line_num} due to invalid label: {parts[1]}") # Use adjusted line num
                except ValueError:
                    print(f"Warning: Skipping line {line_num} due to non-integer label: {parts[1]}") # Use adjusted line num
            else:
                print(f"Warning: Skipping malformed line {line_num}: {line.strip()}") # Use adjusted line num
        print(f"Read {len(examples)} valid examples.")
        return examples
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {filepath}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load or process TSV data: {e}")
        traceback.print_exc()
        return None

def main():
    """Runs the main training loop for ListOps."""

    # --- Dynamic Run Name ---
    dynamics_label = f"Dynamic(T={T},dt={dt},tau={tau_att})" if dynamic else "Static"
    run_name = f"ListOps_{dynamics_label}_L{n_layer}_H{n_head}_E{n_embd}_B{batch_size}_lr{lr}"
    output_dir = os.path.join(output_dir_base, run_name)
    print(f"Run Name: {run_name}")
    print(f"Output Dir: {output_dir}")

    # --- Setup ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device != 'cpu' else nullcontext()
    print(f"Device: {device}, Dtype: {dtype}, Dynamic Mode: {dynamic}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading ---
    train_file = os.path.join(lra_data_root, 'listops-1000', 'basic_train.tsv')
    val_file = os.path.join(lra_data_root, 'listops-1000', 'basic_val.tsv') # Using val set for evaluation during training

    train_data_raw = load_listops_tsv(train_file)
    val_data_raw = load_listops_tsv(val_file)

    if train_data_raw is None or val_data_raw is None:
        print("Failed to load data. Exiting.")
        return

    # --- Model Initialization ---
    model_conf_args = dict(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        block_size=block_size, vocab_size=LISTOPS_VOCAB_SIZE, dropout=dropout,
        tau_att=tau_att, tau_v=tau_v, dt=dt, T=T,
        # IMPORTANT: Pass use_prospective_coding based on your model's needs
        # use_prospective_coding=False # Example for simplified dynamics
    )
    config = GPTConfig(**model_conf_args)
    # Pass num_classes to GPT constructor
    model = GPT(config, num_classes=LISTOPS_NUM_CLASSES)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=optimizer_steps_target, eta_min=lr * 0.1)
    grad_clip = 1.0

    # --- Batching Function for ListOps ---
    data_indices = {'train': list(range(len(train_data_raw))), 'val': list(range(len(val_data_raw)))}

    def get_batch(split: str = 'train') -> tuple[torch.Tensor, torch.Tensor]:
        """Gets a batch of ListOps data, encodes, and pads."""
        data_raw = train_data_raw if split == 'train' else val_data_raw
        idx_pool = data_indices[split]
        if len(idx_pool) < batch_size:
            raise ValueError(f"Dataset split '{split}' has only {len(idx_pool)} samples, less than batch_size {batch_size}")

        # Select random indices for the batch
        current_indices = random.sample(idx_pool, batch_size)
        batch_raw = [data_raw[i] for i in current_indices]

        # Encode the text inputs
        batch_encoded = [{'input': encode_listops_string(ex['input_text'], LISTOPS_VOCAB),
                          'target': ex['target_label']} for ex in batch_raw]

        # Pad sequences within this batch
        max_len_batch = max(len(ex['input']) for ex in batch_encoded)
        # Ensure max length doesn't exceed model block size (truncate if necessary)
        # Note: This simple truncation might cut off crucial info for ListOps
        max_len_batch = min(max_len_batch, block_size)

        padded_inputs = []
        targets = []
        for ex in batch_encoded:
            encoded = ex['input']
            # Truncate if longer than max_len_batch (or block_size)
            if len(encoded) > max_len_batch:
                 encoded = encoded[:max_len_batch]
            # Pad if shorter
            padded_input = np.pad(encoded, (0, max_len_batch - len(encoded)), constant_values=LISTOPS_VOCAB['<pad>'])
            padded_inputs.append(padded_input)
            targets.append(ex['target'])

        # Stack into tensors
        x = torch.tensor(np.stack(padded_inputs), dtype=torch.long)
        y = torch.tensor(np.array(targets), dtype=torch.long) # Target shape (B,)

        return x.to(device), y.to(device)

    # --- Loss Estimation Function ---
    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        """Estimates loss for ListOps classification."""
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_batches)
            correct = 0
            total = 0
            for k in range(eval_batches):
                try:
                    X, Y = get_batch(split)
                    with ctx:
                        # Pass task_type implicitly via model structure (num_classes set)
                        logits, loss = model.step(X, Y) if dynamic else model(X, Y)
                    if loss is not None and not torch.isnan(loss):
                        losses[k] = loss.item()
                        # Calculate accuracy for this batch
                        if logits is not None:
                             pred = torch.argmax(logits, dim=-1)
                             correct += (pred == Y).sum().item()
                             total += Y.size(0)
                    else:
                        losses[k] = float('nan')
                except ValueError:
                    losses[k] = float('nan')

            valid_losses = losses[~torch.isnan(losses)]
            out[f'{split}_loss'] = valid_losses.mean().item() if len(valid_losses) > 0 else float('nan')
            out[f'{split}_acc'] = (correct / total * 100.0) if total > 0 else 0.0
        model.train()
        return out

    # --- Training Loop ---
    print(f"\nStarting ListOps training for {max_iters} iterations...")
    start_time = time.time()
    iter_num = 0
    optimizer_steps = 0
    best_val_acc = -1.0 # Track best accuracy
    running_train_loss = 0.0

    # Shuffle training indices at the start of each "epoch" equivalent
    random.shuffle(data_indices['train'])
    data_idx_pos = 0

    try:
        while iter_num < max_iters: # iter_num now tracks micro-batch iterations

            # --- Evaluation ---
            # Evaluate based on optimizer_steps or iter_num if you prefer more frequent eval
            # Here, we evaluate based on effective full batches processed
            if (iter_num // accumulation_steps) % (eval_interval // accumulation_steps if accumulation_steps > 0 else eval_interval) == 0 and iter_num > 0 or iter_num == max_iters - 1 :
                if iter_num % accumulation_steps == 0: # Ensure we evaluate after an optimizer step
                    metrics = estimate_loss()
                    val_loss = metrics.get('val_loss', float('inf'))
                    val_acc = metrics.get('val_acc', 0.0)
                    print(f"Iter {iter_num} (Opt Step {optimizer_steps}): Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        # ... (save model logic) ...
                        checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'config': config,
                            'optimizer_step': optimizer_steps, # Save optimizer step
                            'best_val_acc': best_val_acc,
                            'dynamic': dynamic
                        }
                        ckpt_path = os.path.join(output_dir, 'best_model.pt')
                        print(f"Saving best model with val_acc {best_val_acc:.2f}% to {ckpt_path}")
                        torch.save(checkpoint, ckpt_path)


            # --- Training Step (Micro-batch) ---
            t0 = time.time()
            try:
                X, Y = get_batch('train')
            except ValueError as e:
                 print(f"Warning: Reshuffling training data at iter {iter_num}: {e}")
                 random.shuffle(data_indices['train'])
                 if len(data_indices['train']) < batch_size:
                     print("ERROR: Not enough training data to form a batch even after reshuffle.")
                     break # Exit training loop
                 continue

            current_actual_lr = optimizer.param_groups[0]['lr'] # Get LR before optimizer step
            with ctx:
                logits, loss = model.step(X, Y) if dynamic else model(X, Y)

            if loss is not None and not torch.isnan(loss):
                loss_item_unscaled = loss.item() # For accurate accumulation for logging
                running_train_loss += loss_item_unscaled

                # --- Scale loss for accumulation ---
                loss = loss / accumulation_steps
                # -----------------------------------

                # --- Accumulate gradients ---
                loss.backward() # Gradients are summed up on .grad attributes
                # -------------------------

                if grad_clip > 0:
                    # It's often better to clip accumulated gradients, but clipping per micro-batch is also common.
                    # If clipping accumulated, do it only before optimizer.step().
                    # For simplicity here, let's clip per micro-batch if grad_clip is small.
                    # If grad_clip is large (like 1.0), total norm can grow with accumulation_steps.
                    # A more robust way is to unscale grads, clip, then scale back, or clip total norm.
                    # For now, this simple clipping is fine.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                # --- Step optimizer and clear grads only after N accumulation_steps ---
                if (iter_num + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step() # Step scheduler with the optimizer
                    optimizer_steps += 1
                # ------------------------------------------------------------------
            else:
                loss_item_unscaled = -1.0
                print(f"Warning: Skipping optimizer step {iter_num} due to invalid loss.")


            # --- Logging ---
            # Log based on micro-batch iterations, but average over actual accumulation window
            if (iter_num + 1) % (log_interval * accumulation_steps) == 0 and iter_num > 0:
                 avg_train_loss = running_train_loss / (log_interval * accumulation_steps) if loss_item_unscaled != -1.0 else -1.0
                 print(f"Iter {iter_num} (Opt Step {optimizer_steps}): Avg Train Loss = {avg_train_loss:.4f}, LR = {current_actual_lr:.6f}")
                 running_train_loss = 0.0 # Reset accumulator

            t1 = time.time()
            if (iter_num + 1) % (100 * accumulation_steps) == 0 and iter_num > 0: # Log step time per effective batch
                 # Calculate time for the last 'accumulation_steps' micro-batches
                 # This is a bit tricky to get an exact "effective batch time" here easily
                 # The current (t1-t0) is for one micro-batch.
                 print(f"Iter {iter_num} (Opt Step {optimizer_steps}): micro-batch time {(t1-t0)*1000:.2f}ms")

            iter_num += 1
            if optimizer_steps >= (max_iters // accumulation_steps): # Ensure we don't exceed effective max_iters
                 print(f"Reached effective max_iters based on optimizer steps ({optimizer_steps}). Stopping.")
                 break

    # --- End of Training Loop ---
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during training: {e}")
        traceback.print_exc()

    # --- Finish Training ---
    print("\nTraining finished.")
    elapsed_time = time.time() - start_time
    print(f"Total training time: {elapsed_time:.2f}s")

    # --- Save Final Model ---
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config, # Save config object
        'final_val_acc': metrics.get('val_acc', 0.0) if 'metrics' in locals() else -1.0,
        'best_val_acc': best_val_acc,
        'dynamic': dynamic
    }
    final_ckpt_path = os.path.join(output_dir, 'final_model.pt')
    print(f"Saving final model checkpoint to {final_ckpt_path}")
    try:
        torch.save(final_checkpoint, final_ckpt_path)
    except Exception as e:
        print(f"ERROR: Failed to save final model: {e}")

if __name__ == "__main__":
    main()