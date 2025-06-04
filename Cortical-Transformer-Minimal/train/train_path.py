from contextlib import nullcontext
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import time
import os
import traceback
import random
import math
import matplotlib.pyplot as plt

""" Training script for a GPT model on LRA Pathfinder, using preprocessed .npy files. """

try:
    from model1 import GPTConfig, GPT
    print("Imported GPTConfig and GPT successfully from model1.py.")
except ImportError:
    print("Error: Failed to import GPTConfig and GPT from model1.py.")
    exit(1)

n_layer: int = 4
n_head: int = 4
n_embd: int = 192
block_size: int = 256
dropout: float = 0.05


PATHFINDER_RESOLUTION: int = 32
LRA_SEQ_LEN: int = PATHFINDER_RESOLUTION * PATHFINDER_RESOLUTION
VOCAB_SIZE: int = 256
NUM_CLASSES: int = 2


dynamic: bool = True
tau_att: float = 1.0
tau_v: float = 1.0
dt: float = 0.01
T: int = 1


batch_size: int = 2
accumulation_steps: int = 16
lr: float = 1e-5
weight_decay: float = 0.03
max_iters: int = 3000
eval_interval: int = 100
eval_batches: int = 50
log_interval: int = 10


seed: int = 1337

PROCESSED_DATA_DIR: str = f'./lra_pathfinder_preprocessed/'
output_dir_base: str = f'./lra_pathfinder{PATHFINDER_RESOLUTION}_output_from_npy'


class NpyPathfinderDataset(Dataset):
    def __init__(self, resolution: int, split_name: str, base_dir: str):
        self.x_path = os.path.join(base_dir, f"pathfinder{resolution}_{split_name}_X.npy")
        self.y_path = os.path.join(base_dir, f"pathfinder{resolution}_{split_name}_Y.npy")
        print(f"Attempting to load X data for {split_name} from: {self.x_path}")
        self.x_data = np.load(self.x_path, mmap_mode='r')
        print(f"Attempting to load Y data for {split_name} from: {self.y_path}")
        self.y_data = np.load(self.y_path, mmap_mode='r')
        assert len(self.x_data) == len(self.y_data), \
            f"X and Y data must have same number of samples. X: {len(self.x_data)}, Y: {len(self.y_data)}"
        print(f"Loaded {len(self.y_data)} samples for Pathfinder-{resolution} {split_name} split.")

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        sample_x = torch.from_numpy(self.x_data[idx].astype(np.int64))
        sample_y = torch.tensor(self.y_data[idx].astype(np.int64), dtype=torch.long)
        return {'input': sample_x, 'target': sample_y}

def main():
   
    dynamics_label = f"Dynamic(T={T},dt={dt},tau={tau_att})" if dynamic else "Static"
    effective_bs = batch_size * accumulation_steps
    run_name = f"LRAPathfinder{PATHFINDER_RESOLUTION}_{dynamics_label}_L{n_layer}_H{n_head}_E{n_embd}_B{effective_bs}_lr{lr}_block{block_size}"
    output_dir = os.path.join(output_dir_base, run_name)
    print(f"Run Name: {run_name}\nOutput Dir: {output_dir}")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device != 'cpu' else nullcontext()
    print(f"Device: {device}, Dtype: {dtype}, Dynamic Mode: {dynamic}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        train_dataset = NpyPathfinderDataset(PATHFINDER_RESOLUTION, 'train', PROCESSED_DATA_DIR)
        val_dataset = NpyPathfinderDataset(PATHFINDER_RESOLUTION, 'validation', PROCESSED_DATA_DIR)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        print("PyTorch DataLoaders created from .npy files.")
    except FileNotFoundError as e:
        print(f"ERROR: Could not load .npy files. Did you run preprocess_pathfinder.py successfully first?")
        print(f"Missing file: {e.filename}")
        print(f"Expected data in directory: {PROCESSED_DATA_DIR}"); return
    except Exception as e:
        print(f"ERROR: Failed to create DataLoaders: {e}"); traceback.print_exc(); return

    model_conf_args = dict(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        block_size=block_size, vocab_size=VOCAB_SIZE, dropout=dropout,
        tau_att=tau_att, tau_v=tau_v, dt=dt, T=T
    )
    config = GPTConfig(**model_conf_args)
    model = GPT(config, num_classes=NUM_CLASSES)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(f"Model config used: {config}")

    try: optimizer = model.configure_optimizers(weight_decay=weight_decay,learning_rate=lr, betas=(0.9,0.95),device_type=device)
    except AttributeError: optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9,0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=lr * 0.1)
    grad_clip = 1.0

    avg_train_losses_plot = []
    val_losses_plot = []
    val_accs_plot = []
    optimizer_steps_plot = []

    def get_pytorch_batch_from_loader(iterator):
        try: batch_data = next(iterator)
        except StopIteration: return None, None
        x = batch_data['input'].to(device)
        y = batch_data['target'].to(device)
        return x,y

    @torch.no_grad()
    def estimate_loss(current_val_data_loader) -> dict[str, float]:
        out = {}; model.eval()
        val_iter_for_eval = iter(current_val_data_loader)
        losses_val, correct_val, total_val, batches_done_eval = torch.zeros(eval_batches), 0, 0, 0
        for k in range(eval_batches):
            batch_data = get_pytorch_batch_from_loader(val_iter_for_eval)
            if batch_data[0] is None: print(f"Val iter exhausted at batch {k} for estimate_loss"); break
            X_full, Y_true = batch_data; batches_done_eval +=1
            final_chunk_logits = None
            current_model_block_size = model.config.block_size
            num_chunks = math.ceil(X_full.size(1) / current_model_block_size)
            for i in range(num_chunks):
                start_idx = i * current_model_block_size
                end_idx = min((i + 1) * current_model_block_size, X_full.size(1))
                X_chunk = X_full[:, start_idx:end_idx]
                with ctx:
                    logits_chunk, _ = model.step(X_chunk, targets=None) if dynamic else model(X_chunk, targets=None)
                if i == num_chunks - 1: final_chunk_logits = logits_chunk
            if final_chunk_logits is not None:
                chunk_loss = F.cross_entropy(final_chunk_logits.view(-1, NUM_CLASSES), Y_true.view(-1))
                losses_val[k] = chunk_loss.item()
                pred = torch.argmax(final_chunk_logits, dim=-1)
                correct_val += (pred == Y_true).sum().item(); total_val += Y_true.size(0)
            else: losses_val[k] = float('nan')
        if batches_done_eval == 0 and k > 0 :
             print(f"Warning: No batches processed in estimate_loss for {eval_batches} attempts.")
        if batches_done_eval < eval_batches and batches_done_eval > 0 : losses_val = losses_val[:batches_done_eval]
        
        valid_losses_val = losses_val[~torch.isnan(losses_val)]
        out['val_loss'] = valid_losses_val.mean().item() if len(valid_losses_val) > 0 else float('inf')
        out['val_acc'] = (correct_val / total_val * 100.0) if total_val > 0 else 0.0
        model.train(); return out

    print(f"\nStarting LRA Pathfinder training for {max_iters} optimizer steps...")
    start_time = time.time()
    micro_batch_iter_num, optimizer_steps, best_val_acc = 0, 0, -1.0
    running_train_loss_sum_for_log = 0.0
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_dataloader)

    try:
        while optimizer_steps < max_iters:
            if optimizer_steps > 0 and optimizer_steps % eval_interval == 0:
                metrics = estimate_loss(val_dataloader)
                val_loss, val_acc = metrics.get('val_loss', float('inf')), metrics.get('val_acc', 0.0)
                current_lr_log = optimizer.param_groups[0]['lr']
                print(f"Opt Step {optimizer_steps}/{max_iters}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, LR={current_lr_log:.2e}")
                val_losses_plot.append(val_loss); val_accs_plot.append(val_acc); optimizer_steps_plot.append(optimizer_steps) # Log for plotting
                if val_acc > best_val_acc and val_acc > 0 :
                    best_val_acc = val_acc
                    checkpoint = {'model_state_dict': model.state_dict(), 'config': config, 'optimizer_step': optimizer_steps, 'best_val_acc': best_val_acc, 'dynamic': dynamic}
                    ckpt_path = os.path.join(output_dir, f'best_model_step{optimizer_steps}.pt'); print(f"Saving best model with val_acc {best_val_acc:.2f}% to {ckpt_path}"); torch.save(checkpoint, ckpt_path)

            t0 = time.time()
            try: batch_data = next(train_iter)
            except StopIteration:
                print(f"Opt Step {optimizer_steps}: Training epoch finished. Re-initializing DataLoader.");
                train_iter = iter(train_dataloader); batch_data = next(train_iter)
            
            X_full, Y_true = batch_data['input'].to(device), batch_data['target'].to(device)
            if X_full is None: print("ERROR: Failed to get batch after re-init DataLoader."); break
            
            final_chunk_logits_train, loss_for_step = None, None
            current_model_block_size = model.config.block_size
            num_chunks = math.ceil(X_full.size(1) / current_model_block_size)
            for i in range(num_chunks):
                start_idx, end_idx = i*current_model_block_size, min((i+1)*current_model_block_size, X_full.size(1))
                X_chunk = X_full[:, start_idx:end_idx]
                with ctx:
                    logits_chunk, _ = model.step(X_chunk, targets=None) if dynamic else model(X_chunk, targets=None)
                if i == num_chunks - 1: final_chunk_logits_train = logits_chunk
            if final_chunk_logits_train is not None:
                loss_for_step = F.cross_entropy(final_chunk_logits_train.view(-1, NUM_CLASSES), Y_true.view(-1))

            if loss_for_step is not None and not torch.isnan(loss_for_step):
                loss_item_unscaled = loss_for_step.item(); running_train_loss_sum_for_log += loss_item_unscaled
                loss_for_step = loss_for_step / accumulation_steps
                loss_for_step.backward()
                if (micro_batch_iter_num + 1) % accumulation_steps == 0:
                    if grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
                    optimizer_steps += 1
                    if optimizer_steps % log_interval == 0 and optimizer_steps > 0:
                        avg_train_loss = running_train_loss_sum_for_log / (log_interval * accumulation_steps)
                        current_lr_log = optimizer.param_groups[0]['lr']
                        print(f"Opt Step {optimizer_steps}: Avg Train Loss = {avg_train_loss:.4f}, LR = {current_lr_log:.2e}")
                        avg_train_losses_plot.append(avg_train_loss)
                        running_train_loss_sum_for_log = 0.0
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
        final_checkpoint = {'model_state_dict': model.state_dict(), 'config': config, 'optimizer_step': optimizer_steps, 'best_val_acc': best_val_acc, 'dynamic': dynamic}
        ckpt_path = os.path.join(output_dir, 'final_model.pt'); print(f"Saving final model to {ckpt_path}"); torch.save(final_checkpoint, ckpt_path)

        
        if optimizer_steps_plot and val_losses_plot and avg_train_losses_plot:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            color = 'tab:blue'
            ax1.set_xlabel('Optimizer Steps')
            ax1.set_ylabel('Loss', color=color)
            train_log_optimizer_steps = [i for i in range(log_interval, optimizer_steps + 1, log_interval)]
            ax1.plot(train_log_optimizer_steps[:len(avg_train_losses_plot)], avg_train_losses_plot, color=color, linestyle=':', label='Avg Train Loss')
            ax1.plot(optimizer_steps_plot, val_losses_plot, color=color, linestyle='-', marker='o', label='Validation Loss')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, axis='y', linestyle=':')

            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('Accuracy (%)', color=color)
            ax2.plot(optimizer_steps_plot, val_accs_plot, color=color, linestyle='--', marker='x', label='Validation Accuracy')
            ax2.tick_params(axis='y', labelcolor=color)
            
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='best')
            fig.suptitle(f'Training Progress ({run_name})')
            fig.tight_layout()
            plot_path = os.path.join(output_dir, 'training_plot.png'); print(f"Saving training plot to {plot_path}"); plt.savefig(plot_path); plt.close(fig)
        else: print("Not enough data collected to generate plots.")

if __name__ == "__main__":
    main()