"""
experiments/basic_retrieval.py (Pure PyTorch Version)

Test 1: Basic Memory Retrieval.
Writes N memories, then queries one of them.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from dataset.generator import generate_orthogonal_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity
from adapters.existing_memorybank import MemoryBankAdapter


def run_basic_retrieval(adapter, num_memories, dim, seed):
    """
    Test 1: Basic Memory Retrieval
    Writes N memories, then queries one of them.
    """
    dataset = generate_orthogonal_dataset(seed, num_memories, dim)
    h_eos, is_eos, write_prob, _ = create_synthetic_batch(dataset)

    for i in range(num_memories):
        adapter.write_only(h_eos[i:i+1], is_eos[i:i+1], write_prob[i:i+1])

    target_idx = num_memories // 2
    query_h_eos = h_eos[target_idx:target_idx+1]

    retrieved_v = adapter.read_only(query_h_eos)
    expected_v = adapter.get_v_proj(query_h_eos)

    sim = compute_cosine_similarity(retrieved_v, expected_v)
    return float(np.mean(sim))


def run_experiment(adapter=None, config=None, seeds=5):
    print("Running Basic Retrieval Test (PyTorch)...")
    if adapter is None:
        adapter = MemoryBankAdapter()
        adapter.setup()
    dims = adapter.config.memory_dim
    sizes = [10, 50, 100]
    results = {}
    for size in sizes:
        scores = []
        for seed in range(seeds):
            adapter.reset_memory()
            score = run_basic_retrieval(adapter, size, dims, seed=seed + 300)
            scores.append(score)
        results[size] = {'mean': float(np.mean(scores)), 'std': float(np.std(scores))}
        print(f"  Size {size:3d}: Sim = {results[size]['mean']:.4f} ± {results[size]['std']:.4f}")
    return results


if __name__ == '__main__':
    run_experiment()
