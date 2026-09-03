"""
scripts/generate_dataset.py (Pure PyTorch Version)
"""
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.generator import generate_random_dataset, generate_orthogonal_dataset


def main():
    print("Generating synthetic offline datasets for extended tests (if needed)...")
    seed = 42

    dim = 128
    num_samples = 1000

    dataset = generate_random_dataset(seed, num_samples, dim)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "synthetic_data.npy")

    np.save(out_file, dataset.cpu().numpy())
    print(f"Saved synthetic dataset with shape {dataset.shape} to {out_file}")


if __name__ == "__main__":
    main()
