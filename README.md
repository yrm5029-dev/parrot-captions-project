# Parrot Captions Teach CLIP to Spot Text - Course Project

This repository contains the replication and improvement code for the Course Project based on the paper **"Parrot Captions Teach CLIP to Spot Text" (ECCV 2024)**.

## Project Structure
- `eval_bias.py`: Main script to replicate valid bias scores and run improvement experiments.
- `sys_benchmark.py`: Helper class for generating synthetic benchmark images (modified for Pillow 10+ compatibility).
- `kmeans.py`: Original utility from the source repo (unused in replication).
- `requirements.txt`: List of dependencies.
- `report.md`: Course Project Report draft.

## Setup
1. Clone the repository (or ensure you have the files).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: You need `torch` and `torchvision` tailored to your system, e.g., CPU or CUDA).

# How to verify in 15 minutes

1.  **Install**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run quick baseline eval (CPU)**:
    ```bash
    python eval_bias.py --model laion2b_s34b_b79k --nouns nouns.txt --batch_size 8 --seed 42 --device cpu --output_csv results/baseline_eval.csv
    ```

3.  **Run quick adapter train**:
    ```bash
    python train_adapter.py --subset 200 --epochs 2 --batch_size 8 --seed 42 --output_dir results/adapter_quick
    ```

4.  **Check outputs**:
    - `results/baseline_eval.summary.txt`
    - `results/adapter_quick/adapter_final.pth`

## Full Replication (Heavy)
- **Adapter Training**: `python train_adapter.py` (Trains on 50k images).
- **Evaluation**: `python eval_bias.py` (Runs full test set).

## Results
- **Standard CLIP Bias**: ~0.32 - 0.35 Sync Score (Replicated).
- **Improved Prompting**: ~0.29 Sync Score (Using "A realistic photo of a {}").

## Reference
Original Paper: [https://arxiv.org/abs/2312.14232](https://arxiv.org/abs/2312.14232)
Original Repo: [https://github.com/opendatalab/CLIP-Parrot-Bias](https://github.com/opendatalab/CLIP-Parrot-Bias)
