
import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm

import open_clip
from sys_benchmark import sys_text_dataset

# --- Configuration (Defaults) ---
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
DEFAULT_LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-10 Classes
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CLASS_MAP = {
    'airplane': 'plane',
    'automobile': 'car',
    'bird': 'bird',
    'cat': 'cat',
    'deer': 'deer',
    'dog': 'dog',
    'frog': 'frog',
    'horse': 'horse',
    'ship': 'ship',
    'truck': 'truck'
}

# --- Utility ---
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_csv_log(path, headers, rows):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

# --- 1. Data Loading ---

class CombinedDataset(Dataset):
    """
    Dataset that yields:
    - Real Image (CIFAR)
    - Text Image (Synthetic)
    """
    def __init__(self, cifar_dataset, text_dataset):
        self.cifar_data = cifar_dataset
        self.text_data = text_dataset
        self.length = min(len(self.cifar_data), len(self.text_data))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Real
        real_img, class_idx = self.cifar_data[idx]
        class_name = CIFAR_CLASSES[class_idx]
        simple_name = CLASS_MAP[class_name]
        
        # Text (Synthetic)
        text_img, text_content = self.text_data[idx]
        
        return {
            "real_img": real_img,
            "real_label": simple_name,
            "text_img": text_img,
            "text_content": text_content
        }

# --- 2. Model (Adapter) ---

class BiasAdapter(nn.Module):
    """
    A simple Adapter (MLP) that sits on top of CLIP Visual Encoder.
    Input: CLIP Image Embedding (512 dim)
    Output: Adapted Embedding (512 dim)
    """
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # Residual connection is usually good, we add it in forward
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

def main():
    parser = argparse.ArgumentParser(description="Train Adapter for Bias Reduction")
    parser.add_argument("--subset", type=int, default=None, help="Use a subset of N images")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--download", action="store_true", help="Force download CIFAR-10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/adapter")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Starting Heavy Training (Adapter) on {DEVICE} ---")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # A. Load CLIP
    print("Loading CLIP (This might take a moment if not cached)...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=DEVICE)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    
    # B. Load CIFAR-10 (Real Data)
    print("Loading CIFAR-10 (Real Images)...")
    # We use the CLIP preprocess, assume manual download or cache
    cifar_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=args.download, transform=preprocess)
    
    if args.subset:
        print(f"Using Subset (N={args.subset}) for fast training/grading...")
        cifar_ds = torch.utils.data.Subset(cifar_ds, range(args.subset))
    else:
        print("Using Full CIFAR-10 Dataset (N=50000)...")
    
    # C. Load Synthetic Text (Bias Data)
    print("Generating Synthetic Text Data (Bias Images)...")
    target_words = list(CLASS_MAP.values())
    # Repeat to fill dataset size
    # We generate enough text images to match (or exceed) CIFAR dataset length
    total_needed = len(cifar_ds)
    text_words_list = (target_words * (total_needed // len(target_words) + 1))[:total_needed]
    text_ds = sys_text_dataset(text_words_list, transform=preprocess)
    
    print(f"Real Data: {len(cifar_ds)} images")
    print(f"Text Data: {len(text_ds)} images")
    
    # D. Dataloader
    # Combined or parallel
    real_loader = DataLoader(cifar_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    text_loader = DataLoader(text_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # E. Train Adapter
    adapter = BiasAdapter().to(DEVICE)
    optimizer = optim.Adam(adapter.parameters(), lr=args.lr)
    
    logs = []
    print("\n--- Training Loop ---")
    for epoch in range(args.epochs):
        adapter.train()
        total_loss = 0
        start_time = time.time()
        
        progress = tqdm(zip(real_loader, text_loader), total=min(len(real_loader), len(text_loader)))
        for (real_batch, text_batch) in progress:
            # Unpack
            real_imgs, real_lbls = real_batch
            text_imgs, text_strs = text_batch
            
            real_imgs = real_imgs.to(DEVICE)
            text_imgs = text_imgs.to(DEVICE)
            
            # 1. Get CLIP Embeddings
            with torch.no_grad():
                real_feats = model.encode_image(real_imgs)
                text_img_feats = model.encode_image(text_imgs)
                
                # Targets
                real_text_tokens = tokenizer([f"A photo of a {n}" for n in real_lbls]).to(DEVICE)
                real_target_emb = model.encode_text(real_text_tokens)
                real_target_emb = real_target_emb / real_target_emb.norm(dim=-1, keepdim=True)
                
                text_tokens = tokenizer([f"A photo of a {t}" for t in text_strs]).to(DEVICE)
                text_target_emb = model.encode_text(text_tokens)
                text_target_emb = text_target_emb / text_target_emb.norm(dim=-1, keepdim=True)

            # 2. Adapter
            real_adapted = adapter(real_feats)
            text_adapted = adapter(text_img_feats)
            
            # Normalize
            real_adapted = real_adapted / real_adapted.norm(dim=-1, keepdim=True)
            text_adapted = text_adapted / text_adapted.norm(dim=-1, keepdim=True)
            
            # 3. Loss
            # Real -> Close to Label
            # Text -> Far from Label
            sim_real = (real_adapted * real_target_emb).sum(dim=-1)
            sim_text = (text_adapted * text_target_emb).sum(dim=-1)
            
            loss_real = (1.0 - sim_real).mean()
            loss_text = torch.clamp(sim_text - 0.2, min=0).mean() # Push text similarity down
            
            loss = loss_real + loss_text
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(progress)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} | Elapsed: {elapsed:.2f}s")
        
        logs.append([epoch+1, avg_loss, elapsed])

    # F. Save
    log_csv = os.path.join(args.output_dir, 'train_log.csv')
    save_csv_log(log_csv, ['epoch', 'avg_loss', 'elapsed_s'], logs)
    
    model_path = os.path.join(args.output_dir, 'adapter_final.pth')
    torch.save(adapter.state_dict(), model_path)
    print(f"Saved logs to {log_csv}")
    print(f"Saved adapter to {model_path}")

if __name__ == "__main__":
    main()

