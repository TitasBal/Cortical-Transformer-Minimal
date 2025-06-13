# train_lra_image.py

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
    tf.config.experimental.set_visible_devices([], 'GPU')
    TFDS_AVAILABLE = True
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow Datasets version: {tfds.__version__}")
except ImportError:
    print("CRITICAL ERROR: tensorflow-datasets or tensorflow is not installed.")
    TFDS_AVAILABLE = False
    exit(1)

""" Training script for a GPT model on LRA Image Classification (CIFAR-10 as sequence). """

try:
    from model1 import GPTConfig, GPT # Or model1 if that's your filename
    print("Imported GPTConfig and GPT successfully.")
except ImportError:
    print("Error: Failed to import GPTConfig and GPT from model_dynamic.py (or model1.py).")
    exit(1)

# --- Configuration ---
# Model Args
n_layer: int = 8
n_head: int = 8
n_embd: int = 512
block_size: int =  1024   # Model's internal processing chunk size (3072 is full length)
                           # Start smaller, e.g., 256 or 512, due to sequence length
dropout: float = 0.1

# --- LRA Image (CIFAR-10) Specific Config ---
LRA_TASK_NAME: str = 'cifar10'
LRA_SEQ_LEN: int = 32 * 32 * 3 # 3072
VOCAB_SIZE: int = 256     # Pixel values 0-255
NUM_CLASSES: int = 10     # CIFAR-10 classes

# Dynamic Args
dynamic: bool = True      # <<< Set to True to use model.step()
tau_att: float = 1.0
tau_v: float = 1.0
dt: float = 0.01
T: int = 1
#use_prospective_coding: bool = False # If your model uses this flag in GPTConfig

# Training Args
batch_size: int = 1          # Physical batch size (adjust based on memory)
accumulation_steps: int = 1   # Effective batch size = 16 * 2 = 32
lr: float = 1e-4              # Learning rate
weight_decay: float = 0.1
max_iters: int = 1500        # Number of OPTIMIZER STEPS (CIFAR-10 has 50k train images)
                               # 50000 / 32_eff_batch = ~1562 steps/epoch. 20k steps = ~12 epochs
eval_interval: int = 500      # Evaluate every N OPTIMIZER STEPS
eval_batches: int = 100       # Batches for validation metric estimation
log_interval: int = 50        # Log train loss every N OPTIMIZER STEPS

# Other
seed: int = 1337
data_dir: str = os.path.expanduser('~/tensorflow_datasets') # TFDS download/cache directory
output_dir_base: str = './lra_image_cifar10_output'
# --- End Configuration ---

def main():
    """Runs the main training loop for LRA Image Classification."""

    # --- Dynamic Run Name ---
    #prospective_label = "Prospective" if use_prospective_coding and dynamic else "Simplified"
    dynamics_label = f"Dynamic_(T={T},dt={dt},tau={tau_att})" if dynamic else "Static"
    effective_bs = batch_size * accumulation_steps
    run_name = f"LRAImage_{dynamics_label}_L{n_layer}_H{n_head}_E{n_embd}_B{effective_bs}_lr{lr}_block{block_size}"
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
    #if dynamic: print(f"Prospective Coding: {use_prospective_coding}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading (Using TFDS for CIFAR-10) ---
    print(f"Loading {LRA_TASK_NAME} from TFDS (data_dir: {data_dir})...")
    try:
        # CIFAR-10: TFDS 'train' split has 50k, 'test' has 10k.
        # We'll use a portion of 'train' for our validation during training.
        train_split_str = 'train[:90%]' # e.g., 45,000 for training
        val_split_str = 'train[90%:]'   # e.g., 5,000 for validation

        @tf.function
        def preprocess_tf_example_cifar(example):
            image = example['image'] # tf.Tensor, shape (32, 32, 3), dtype=uint8
            label = example['label'] # tf.Tensor, shape (), dtype=int64

            # Flatten the image: (32, 32, 3) -> (3072)
            image_flattened = tf.reshape(image, [-1]) # -1 infers the size

            # Ensure sequence length matches LRA_SEQ_LEN (should be 3072 already)
            # No padding/truncation needed if image size is fixed.
            # If LRA_SEQ_LEN was different, you'd pad/truncate here.
            # For CIFAR10 32*32*3 = 3072, so LRA_SEQ_LEN should be 3072.
            # tf.debugging.assert_equal(tf.shape(image_flattened)[0], LRA_SEQ_LEN,
            #                            message="Flattened image length mismatch")

            return image_flattened, label # Return as tuple (input_sequence, label)

        common_load_args = {
            'name': LRA_TASK_NAME, # 'cifar10'
            'data_dir': data_dir,
            'as_supervised': False # We want the dictionary with 'image' and 'label' keys
        }

        print("Loading and preprocessing train data...")
        train_ds_tf = tfds.load(**common_load_args, split=train_split_str, shuffle_files=True)
        train_ds_tf = train_ds_tf.map(preprocess_tf_example_cifar, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds_tf = train_ds_tf.cache()
        train_ds_tf = train_ds_tf.shuffle(buffer_size=10000)
        train_ds_tf = train_ds_tf.batch(batch_size)
        train_ds_tf = train_ds_tf.prefetch(tf.data.AUTOTUNE)
        train_data_iter = train_ds_tf.as_numpy_iterator()

        print("Loading and preprocessing validation data...")
        val_ds_tf = tfds.load(**common_load_args, split=val_split_str, shuffle_files=False)
        val_ds_tf = val_ds_tf.map(preprocess_tf_example_cifar, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds_tf = val_ds_tf.batch(batch_size) # Use same physical batch size for eval consistency
        val_ds_tf = val_ds_tf.prefetch(tf.data.AUTOTUNE)
        # val_data_iter for estimate_loss will be created fresh each time

        print(f"TFDS data pipelines prepared for CIFAR-10.")

    except Exception as e:
        print(f"ERROR: Failed to load or preprocess data from TFDS: {e}")
        traceback.print_exc()
        return

    # --- Model Initialization ---
    model_conf_args = dict(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        block_size=block_size, vocab_size=VOCAB_SIZE, dropout=dropout,
        tau_att=tau_att, tau_v=tau_v, dt=dt, T=T,
        #use_prospective_coding=use_prospective_coding if dynamic else False
    )
    config = GPTConfig(**model_conf_args)
    model = GPT(config, num_classes=NUM_CLASSES)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # --- Optimizer & Scheduler ---
    try:
        optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=lr, betas=(0.9, 0.95), device_type=device)
    except AttributeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=lr * 0.1)
    grad_clip = 1.0

    # --- Helper: Get Batch for PyTorch (from TFDS iterator) ---
    def get_pytorch_batch(iterator):
        try:
            # preprocess_tf_example_cifar returns (image_flattened, label)
            x_np, y_np = next(iterator)
            x = torch.from_numpy(x_np).to(torch.long).to(device) # Pixel values are the "tokens"
            y = torch.from_numpy(y_np).to(torch.long).to(device)
            return x, y
        except StopIteration:
            return None, None

    # --- Loss Estimation Function ---
    @torch.no_grad()
    def estimate_loss(current_val_data_iter) -> dict[str, float]: # Pass val iterator
        out = {}
        model.eval()
        # Only evaluate on validation set during training for speed
        split_name = 'val'
        losses_val = torch.zeros(eval_batches)
        correct_val, total_val = 0, 0
        batches_done_eval = 0

        for k in range(eval_batches):
            X_full, Y_true = get_pytorch_batch(current_val_data_iter)
            if X_full is None: # Val iterator exhausted
                print(f"Validation iterator exhausted early at batch {k} during estimate_loss.")
                break
            batches_done_eval +=1

            final_chunk_logits = None
            num_chunks = math.ceil(X_full.size(1) / block_size) # model.config.block_size
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
        
        if batches_done_eval < eval_batches and batches_done_eval > 0 :
            losses_val = losses_val[:batches_done_eval] # Only average over actual batches

        valid_losses_val = losses_val[~torch.isnan(losses_val)]
        out[f'{split_name}_loss'] = valid_losses_val.mean().item() if len(valid_losses_val) > 0 else float('nan')
        out[f'{split_name}_acc'] = (correct_val / total_val * 100.0) if total_val > 0 else 0.0
        model.train()
        return out

    # --- Training Loop (similar structure, using optimizer_steps) ---
    print(f"\nStarting LRA Image (CIFAR-10 Seq) training for {max_iters} optimizer steps...")
    start_time = time.time()
    micro_batch_iter_num, optimizer_steps, best_val_acc = 0, 0, -1.0
    running_train_loss_sum = 0.0
    optimizer.zero_grad(set_to_none=True)

    try:
        while optimizer_steps < max_iters:
            if optimizer_steps > 0 and optimizer_steps % eval_interval == 0:
                val_iter_for_eval = val_ds_tf.as_numpy_iterator() # Fresh val iterator
                metrics = estimate_loss(val_iter_for_eval)
                val_loss = metrics.get('val_loss', float('inf'))
                val_acc = metrics.get('val_acc', 0.0)
                current_lr_log = optimizer.param_groups[0]['lr']
                print(f"Opt Step {optimizer_steps}/{max_iters}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, LR={current_lr_log:.2e}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # ... (save model logic) ...
                    checkpoint = {'model_state_dict': model.state_dict(), 'config': config,
                                  'optimizer_step': optimizer_steps, 'best_val_acc': best_val_acc,
                                  'dynamic': dynamic}
                    ckpt_path = os.path.join(output_dir, f'best_model_step{optimizer_steps}.pt')
                    print(f"Saving best model with val_acc {best_val_acc:.2f}% to {ckpt_path}")
                    torch.save(checkpoint, ckpt_path)

            t0 = time.time()
            X_full, Y_true = get_pytorch_batch(train_data_iter)
            if X_full is None: # Epoch finished
                print(f"Opt Step {optimizer_steps}: Training epoch finished. Re-initializing DataLoader.")
                train_data_iter = train_ds_tf.as_numpy_iterator()
                X_full, Y_true = get_pytorch_batch(train_data_iter)
                if X_full is None: print("ERROR: Failed to get batch after re-init."); break

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
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
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
        # ... (save final model) ...
        final_checkpoint = {'model_state_dict': model.state_dict(), 'config': config,
                            'optimizer_step': optimizer_steps, 'best_val_acc': best_val_acc,
                            'dynamic': dynamic }
        ckpt_path = os.path.join(output_dir, 'final_model.pt')
        print(f"Saving final model to {ckpt_path}")
        torch.save(final_checkpoint, ckpt_path)

if __name__ == "__main__":
    main()