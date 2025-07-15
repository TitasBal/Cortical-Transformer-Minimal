import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms
import time
import os
import random
import math


try:
    from model2 import GPTConfig, GPT
    print("Imported GPTConfig and GPT successfully from model2.py.")
except ImportError:
    print("Error: Failed to import GPTConfig and GPT from model2.py.")
    exit(1)

n_layer: int = 4
n_head: int = 8
n_embd: int = 128
dropout: float = 0

PATHFINDER_RESOLUTION: int = 32
LRA_SEQ_LEN: int = PATHFINDER_RESOLUTION * PATHFINDER_RESOLUTION
block_size: int = LRA_SEQ_LEN
VOCAB_SIZE: int = 256
NUM_CLASSES: int = 2

learning_rate: float = 3e-4
batch_size: int = 32
accumulation_steps: int = 1
max_iters: int = 15000
eval_interval: int = 250
eval_batches: int = 100
log_interval: int = 20

warmup_iters: int = 500
weight_decay: float = 0.1
grad_clip: float = 1.0

seed: int = 1337
PROCESSED_DATA_DIR: str = f'./lra_pathfinder_preprocessed/'
output_dir_base: str = f'./lra_2d_pe_tuning_runs'

class NpyPathfinderDataset(Dataset):
    def __init__(self, split_name: str, base_dir: str, augment: bool = False):
        self.x_path = os.path.join(base_dir, f"pathfinder{PATHFINDER_RESOLUTION}_{split_name}_X.npy")
        self.y_path = os.path.join(base_dir, f"pathfinder{PATHFINDER_RESOLUTION}_{split_name}_Y.npy")
        self.x_data = np.load(self.x_path, mmap_mode='r')
        self.y_data = np.load(self.y_path, mmap_mode='r')
        self.augment = augment
        if self.augment:
            self.transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor()])
        print(f"Loaded {len(self.y_data)} samples for {split_name} split. Augmentation: {self.augment}")
    def __len__(self): return len(self.y_data)
    def __getitem__(self, idx):
        image_2d = self.x_data[idx].reshape(PATHFINDER_RESOLUTION, PATHFINDER_RESOLUTION).astype(np.uint8)
        label = torch.tensor(self.y_data[idx].astype(np.int64), dtype=torch.long)
        if self.augment:
            image_tensor = self.transform(image_2d)
            image_flat = (image_tensor.squeeze() * 255).long().flatten()
            return image_flat, label
        else:
            return torch.from_numpy(self.x_data[idx].astype(np.int64)), label

def get_lr(it):
    if it < warmup_iters: return learning_rate * it / warmup_iters
    if it > max_iters: return learning_rate * 0.1
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    min_lr = learning_rate * 0.1
    return min_lr + coeff * (learning_rate - min_lr)

def main():
    effective_bs = batch_size * accumulation_steps
    run_name = f"Tune2DPE_L{n_layer}_E{n_embd}_lr{learning_rate:.0e}"
    output_dir = os.path.join(output_dir_base, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Run Name: {run_name}")

    torch.manual_seed(seed)
    device = 'cuda'
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type=device, dtype=dtype)

    train_loader = DataLoader(NpyPathfinderDataset('train', PROCESSED_DATA_DIR, augment=True), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(NpyPathfinderDataset('validation', PROCESSED_DATA_DIR, augment=False), batch_size=batch_size, num_workers=0, pin_memory=True)

    config = GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, vocab_size=VOCAB_SIZE, dropout=dropout)
    model = GPT(config, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95), fused=True)

    @torch.no_grad()
    def estimate_loss():
        out = {}; model.eval()
        loader = val_loader
        losses = torch.zeros(eval_batches); total_correct, total_samples = 0, 0
        val_iter = iter(loader)
        for k in range(eval_batches):
            try: X, Y = next(val_iter)
            except StopIteration: break
            X, Y = X.to(device), Y.to(device)
            with ctx: logits, loss = model(X, Y)
            losses[k] = loss.item()
            total_correct += (torch.argmax(logits, dim=-1) == Y).sum().item(); total_samples += Y.size(0)
        out['val_loss'] = losses.mean().item(); out['val_acc'] = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0
        model.train(); return out

    print(f"\nStarting training for {max_iters} steps...")
    step, best_val_acc = 0, -1.0; running_loss = 0.0; train_iter = iter(train_loader)
    while step < max_iters:
        lr = get_lr(step); optimizer.param_groups[0]['lr'] = lr

        if step > 0 and step % eval_interval == 0:
            metrics = estimate_loss()
            print(f"Step {step}/{max_iters}: Val Loss={metrics['val_loss']:.4f}, Val Acc={metrics['val_acc']:.2f}%")
            if metrics['val_acc'] > best_val_acc:
                best_val_acc = metrics['val_acc']
                print(f"  -> New best val acc: {best_val_acc:.2f}%. Saving model...")
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))

        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation_steps):
            try: X, Y = next(train_iter)
            except StopIteration: train_iter = iter(train_loader); X, Y = next(train_iter)
            X, Y = X.to(device), Y.to(device)
            with ctx: _, loss = model(X, Y)
            if loss is not None:
                loss = loss / accumulation_steps
                loss.backward(); running_loss += loss.item() * accumulation_steps

        if grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_interval == 0:
            print(f"Step {step}: Avg Train Loss={(running_loss / log_interval):.4f}, LR={lr:.2e}")
            running_loss = 0.0
        step += 1

    print("\nTraining finished.")

if __name__ == "__main__":
    main()
