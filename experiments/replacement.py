"""
experiments/replacement.py – Memory Replacement Experiment (PyTorch Version).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from dataset.generator import generate_orthogonal_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity
from adapters.existing_memorybank import MemoryBankAdapter


def run_replacement(adapter, capacity, dim, over_capacity, seed):
    """
    Test 7: Replacement
    Write C memories. Write over_capacity additional memories. Check if the first ones were replaced.
    """
    initial_dataset = generate_orthogonal_dataset(seed, capacity, dim)
    h_init, is_init, w_init, _ = create_synthetic_batch(initial_dataset)

    new_dataset = generate_orthogonal_dataset(seed + 1, over_capacity, dim)
    h_new, is_new, w_new, _ = create_synthetic_batch(new_dataset)

    adapter.reset_memory()

    # Write initial C memories
    for i in range(capacity):
        adapter.write_only(h_init[i:i+1], is_init[i:i+1], w_init[i:i+1])

    # Read initial memory 0 before replacement
    retrieved_before = adapter.read_only(h_init[0:1])
    sim_before = compute_cosine_similarity(retrieved_before, adapter.get_v_proj(h_init[0:1]))

    # Write additional memories
    for i in range(over_capacity):
        adapter.write_only(h_new[i:i+1], is_new[i:i+1], w_new[i:i+1])

    # Read initial memory 0 after replacement
    retrieved_after = adapter.read_only(h_init[0:1])
    sim_after = compute_cosine_similarity(retrieved_after, adapter.get_v_proj(h_init[0:1]))

    # Count how many of the initial memories are still retrievable
    retrievable_count = 0
    for i in range(capacity):
        r = adapter.read_only(h_init[i:i+1])
        s = compute_cosine_similarity(r, adapter.get_v_proj(h_init[i:i+1]))
        if np.mean(s) > 0.8:
            retrievable_count += 1

    return float(np.mean(sim_before)), float(np.mean(sim_after)), retrievable_count


def run_experiment(adapter=None, config=None, seeds=3):
    print("Running Replacement Test (PyTorch)...")
    if adapter is None:
        adapter = MemoryBankAdapter()
        adapter.setup()
    capacity = adapter.config.memory_capacity
    dims = adapter.config.memory_dim

    over_sizes = [10, capacity // 2, capacity]
    results = {}

    for size in over_sizes:
        scores_before = []
        scores_after = []
        counts = []
        for seed in range(seeds):
            sb, sa, count = run_replacement(adapter, capacity, dims, size, seed=seed + 900)
            scores_before.append(sb)
            scores_after.append(sa)
            counts.append(count)

        results[size] = {
            'before': float(np.mean(scores_before)),
            'after': float(np.mean(scores_after)),
            'retained': float(np.mean(counts)),
        }
        print(f"  Over-capacity {size:3d}: Sim Before={results[size]['before']:.4f}, Sim After={results[size]['after']:.4f}, Retained={results[size]['retained']:.1f}/{capacity}")

    return results


if __name__ == '__main__':
    run_experiment()
