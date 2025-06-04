from contextlib import nullcontext
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
import traceback # For printing error details

""" Training script for a GPT model (static or dynamic).

This script trains a GPT model on a character-level dataset Tiny Shakespeare (input.txt).

Supports both standard static training and dynamic simulation training.
"""


try:
    from old.model import GPTConfig, GPT
except ImportError:
    print("Error: Failed to import GPTConfig and GPT from model.py.")
    exit(1)

# --- Hardcoded Configuration ---
# Model Args
n_layer: int = 4
n_head: int = 4
n_embd: int = 256
block_size: int = 128
dropout: float = 0.1
# Dynamic Args (set dynamic=True to use)
dynamic: bool = True # <<< Set to True to use model.step()
tau_att: float = 2.0
tau_v: float = 2.0 # 20ms
dt: float = 0.1 # 1ms
T: int = 1
# Training Args
batch_size: int = 32
lr: float = 1e-4
weight_decay: float = 0.1
max_iters: int = 5000
eval_interval: int = 100
eval_batches: int = 10
# Other
seed: int = 1337
output_dir: str = './minimal_output'
# --- End Configuration ---

def main():
    """Runs the main training loop."""

    # --- Setup ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Determine appropriate dtype based on availability
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    # Autocast context manager for mixed precision
    ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device != 'cpu' else nullcontext()
    print(f"Device: {device}, Dtype: {dtype}, Dynamic Mode: {dynamic}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading and Preprocessing ---
    data_file = '.\TinyShakespeare\input.txt'
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            text_data = f.read()
    except FileNotFoundError:
        print(f"ERROR: Data file '{data_file}' not found.")
        return

    # Create character vocabulary and encoding/decoding functions
    chars = sorted(list(set(text_data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s if c in stoi] # Simple encoder
    decode = lambda l: ''.join([itos[i] for i in l if i in itos]) # Simple decoder

    # Prepare train/validation splits
    encoded_data = np.array(encode(text_data), dtype=np.int64)
    n = len(encoded_data)
    train_data = encoded_data[:int(n * 0.9)]
    val_data = encoded_data[int(n * 0.9):]
    print(f"Data loaded: {n:,} characters, {vocab_size} vocab size.")

    # --- Model Initialization ---
    model_conf_args = dict(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        block_size=block_size, vocab_size=vocab_size, dropout=dropout,
        tau_att=tau_att, tau_v=tau_v, dt=dt, T=T
    )
    config = GPTConfig(**model_conf_args)
    model = GPT(config)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    # Learning rate decay scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=lr * 0.1)
    grad_clip = 1.0 # Gradient clipping threshold

    # --- Utility Functions ---
    def get_batch(split: str = 'train') -> tuple[torch.Tensor, torch.Tensor]:
        """Selects a random batch of input (x) and target (y) sequences.

        Args:
            split: Which data split to use ('train' or 'val').

        Returns:
            A tuple containing the input and target tensors, moved to the correct device.

        Raises:
            ValueError: If the dataset split is too small for the block size.
        """
        data = train_data if split == 'train' else val_data
        max_ix = len(data) - block_size
        if max_ix < 0: raise ValueError(f"Dataset split '{split}' too small for block_size.")
        # Select random starting indices for the batch
        ix = torch.randint(0, max_ix + 1, (batch_size,))
        # Stack sequences into batch tensors
        x = torch.stack([torch.from_numpy(data[i:i+block_size]) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size]) for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        """Estimates the average loss over several batches for train and val splits.

        Sets the model to evaluation mode and disables gradient calculation.

        Returns:
            A dictionary containing the average loss for 'train' and 'val' splits.
        """
        out = {}
        model.eval() # Set model to evaluation mode
        for split in ['train', 'val']:
            losses = torch.zeros(eval_batches) # Tensor to store losses for averaging
            for k in range(eval_batches):
                try:
                    X, Y = get_batch(split)
                    # Perform forward pass using mixed precision context
                    with ctx:
                        # Use model.step if dynamic=True, otherwise model.forward
                        _, loss = model.step(X, Y) if dynamic else model(X, Y)
                    losses[k] = loss.item() if loss is not None else float('nan')
                except ValueError: # Handle case where split might be too small for all batches
                    losses[k] = float('nan')
            # Calculate mean loss, ignoring potential NaNs
            valid_losses = losses[~torch.isnan(losses)]
            out[split] = valid_losses.mean().item() if len(valid_losses) > 0 else float('nan')
        model.train() # Set model back to training mode
        return out

    # --- Training Loop ---
    print(f"\nStarting training for {max_iters} iterations...")
    start_time = time.time()
    iter_num = 0
    try:
        X, Y = get_batch('train') # Fetch the first batch
    except ValueError as e:
        print(f"ERROR: Failed to get initial batch: {e}")
        return # Cannot start training without data

    try:
        while iter_num < max_iters:
            # --- Evaluation Phase ---
            if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
                losses = estimate_loss()
                val_loss = losses.get('val', float('inf')) # Use inf if val loss couldn't be calculated
                print(f"Iter {iter_num}/{max_iters}: Val Loss = {val_loss:.4f}")

            # --- Training Step ---
            # Forward pass using mixed precision
            with ctx:
                # Use the appropriate model method based on the 'dynamic' flag
                logits, loss = model.step(X, Y) if dynamic else model(X, Y)

            # Backward pass and optimizer step
            if loss is not None and not torch.isnan(loss):
                optimizer.zero_grad(set_to_none=True) # Reset gradients
                loss.backward() # Compute gradients
                if grad_clip > 0: # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step() # Update model parameters
            else:
                # Log a warning if loss is invalid, skip optimization step
                print(f"Warning: Skipping optimizer step at iter {iter_num} due to None/NaN loss.")

            scheduler.step() # Update learning rate

            # --- Fetch Next Batch ---
            try:
                # Prepare data for the next iteration
                X, Y = get_batch('train')
            except ValueError as e:
                print(f"ERROR: Cannot get next batch: {e}")
                return # Stop training if data fetching fails

            iter_num += 1

    except KeyboardInterrupt:
        # Handle user interruption gracefully
        print("\nTraining interrupted by user.")
    except Exception as e:
        # Handle unexpected errors during the loop
        print(f"\nAn unexpected error occurred during training: {e}")
        traceback.print_exc() # Print detailed error information

    # --- Finish Training ---
    print("\nTraining finished.")
    elapsed_time = time.time() - start_time
    print(f"Total training time: {elapsed_time:.2f}s")

    # --- Save Final Model ---
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.__dict__ # Save the configuration used
    }
    final_ckpt_path = os.path.join(output_dir, 'final_model.pt')
    print(f"Saving final model checkpoint to {final_ckpt_path}")
    try:
        torch.save(final_checkpoint, final_ckpt_path)
    except Exception as e:
        print(f"ERROR: Failed to save final model: {e}")

if __name__ == "__main__":
    # Entry point for the script
    main()