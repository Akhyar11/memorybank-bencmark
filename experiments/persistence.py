"""
experiments/persistence.py – Memory Persistence Experiment (PyTorch Version).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from dataset.generator import generate_random_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity
from adapters.existing_memorybank import MemoryBankAdapter


def run_persistence(adapter, num_distractors, dim, seed):
    """
    Test 2: Memory Persistence
    Write target, write N distractors, read target.
    """
    torch.manual_seed(seed)
    target = generate_random_dataset(seed, 1, dim)
    t_eos, t_is_eos, t_w_prob, _ = create_synthetic_batch(target)

    # Write Target
    adapter.write_only(t_eos, t_is_eos, t_w_prob)

    # Generate and Write Distractors
    if num_distractors > 0:
        distractors = generate_random_dataset(seed + 1, num_distractors, dim)
        d_eos, d_is_eos, d_w_prob, _ = create_synthetic_batch(distractors)
        for i in range(num_distractors):
            adapter.write_only(d_eos[i:i+1], d_is_eos[i:i+1], d_w_prob[i:i+1])

    # Read Target
    retrieved_v = adapter.read_only(t_eos)
    expected_v = adapter.get_v_proj(t_eos)

    sim = compute_cosine_similarity(retrieved_v, expected_v)
    return float(np.mean(sim))


def run_experiment(adapter=None, config=None, seeds=3):
    print("Running Persistence Test (PyTorch)...")
    if adapter is None:
        adapter = MemoryBankAdapter()
        adapter.setup()
    dims = adapter.config.memory_dim
    max_distractors = adapter.config.memory_capacity - 2
    sizes = [0, 10, min(100, max_distractors), max_distractors]
    results = {}
    for size in sizes:
        scores = []
        for seed in range(seeds):
            adapter.reset_memory()
            score = run_persistence(adapter, size, dims, seed=seed + 700)
            scores.append(score)
        results[size] = {'mean': float(np.mean(scores)), 'std': float(np.std(scores))}
        print(f"  Distractors {size}: Sim = {results[size]['mean']:.4f} ± {results[size]['std']:.4f}")
    return results


if __name__ == '__main__':
    run_experiment()
