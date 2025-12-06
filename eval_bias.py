import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
# if using open_clip (make sure version pinned in requirements.txt)
import open_clip

from sys_benchmark import sys_text_dataset

# reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_nouns(nouns_path):
    if not os.path.exists(nouns_path):
        # Fallback if file missing
        print(f"Warning: {nouns_path} not found. Using internal list.")
        return ["cat", "dog", "car", "tree", "house"] * 10
    with open(nouns_path, 'r', encoding='utf-8') as f:
        nouns = [l.strip() for l in f.readlines() if l.strip()]
    return nouns

def calculate_sync_score(model, tokenizer, words, prompt_template="{}", device='cpu', batch_size=32):
    """
    Calculates the Sync Score: Average cosine similarity between 
    images containing the text 'word' and the text prompt 'word'.
    Returns: mean, std, list_of_tuples(word, score)
    """
    model.eval()
    
    # Get transform
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    
    dataset = sys_text_dataset(words, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    per_word_results = []
    
    # We need to map back to words. The dataset preserves order.
    word_idx = 0
    
    with torch.no_grad():
        for images, texts in dataloader:
            images = images.to(device)
            
            # Create text prompts
            prompts = [prompt_template.format(t) for t in texts]
            try:
                text_tokens = tokenizer(prompts).to(device)
            except Exception as e:
                # Fallback for old open_clip versions
                text_tokens = open_clip.tokenize(prompts).to(device)
            
            # Encode
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)
            
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute pairwise similarity
            sim = (image_features * text_features).sum(dim=-1).cpu().numpy()
            
            for s, t in zip(sim, texts):
                per_word_results.append((t, float(s)))
            
    scores = [s for _, s in per_word_results]
    return np.mean(scores), np.std(scores), per_word_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='laion2b_s34b_b79k', help='open_clip model name')
    parser.add_argument('--nouns', type=str, default='nouns.txt', help='path to nouns file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu','cuda'])
    parser.add_argument('--output_csv', type=str, default='results/baseline_eval.csv', help='save per-word scores')
    parser.add_argument('--template', type=str, default='A photo of {}', help='text template')
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Ensure results dir exists
    output_path = Path(args.output_csv)
    os.makedirs(output_path.parent, exist_ok=True)

    # load nouns
    nouns = load_nouns(args.nouns)
    print(f"Loaded {len(nouns)} nouns.")

    # device
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {device}")

    # load model
    print(f"Loading Model: {args.model}...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=args.model, device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Run Eval
    print(f"Evaluating with template: '{args.template}'...")
    mean_score, std_score, per_word_results = calculate_sync_score(model, tokenizer, nouns, args.template, device, args.batch_size)
    
    print(f"Aggregate Sync Score: {mean_score:.4f} (+/- {std_score:.4f})")

    # save CSV
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['word', 'score'])
        for w, s in per_word_results:
            writer.writerow([w, s])

    # Save summary file
    summary_path = output_path.with_suffix('.summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"model,{args.model}\n")
        f.write(f"seed,{args.seed}\n")
        f.write(f"device,{device}\n")
        f.write(f"template,{args.template}\n")
        f.write(f"aggregate_sync_score,{mean_score}\n")
        f.write(f"std_dev,{std_score}\n")
        f.write(f"num_words,{len(per_word_results)}\n")
        
    print(f"Saved results to {args.output_csv} and {summary_path}")

if __name__ == '__main__':
    main()
