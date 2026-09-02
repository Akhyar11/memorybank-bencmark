import os
import sys
import jax
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.generator import generate_random_dataset, generate_orthogonal_dataset

def main():
    print("Generating synthetic offline datasets for extended tests (if needed)...")
    key = jax.random.PRNGKey(42)
    
    dim = 128
    num_samples = 1000
    
    dataset = generate_random_dataset(key, num_samples, dim)
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
    out_file = os.path.join(out_dir, "synthetic_data.npy")
    
    np.save(out_file, np.array(dataset))
    print(f"Saved synthetic dataset with shape {dataset.shape} to {out_file}")

if __name__ == "__main__":
    main()
