"""
experiments/interference.py – Interference Experiment (PyTorch Version).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from dataset.generator import generate_random_dataset, generate_interference_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity
from adapters.existing_memorybank import MemoryBankAdapter


def run_interference(adapter, num_distractors, dim, noise_level, seed):
    """
    Test 5: Interference
    Write target, write distractors that are similar to target. Read target.
    """
    torch.manual_seed(seed)
    target = generate_random_dataset(seed, 1, dim)
    t_eos, t_is_eos, t_w_prob, _ = create_synthetic_batch(target)

    # Write Target
    adapter.write_only(t_eos, t_is_eos, t_w_prob)

    # Generate and Write Distractors
    if num_distractors > 0:
        distractors = generate_interference_dataset(seed + 1, target, num_distractors, noise_level)
        d_eos, d_is_eos, d_w_prob, _ = create_synthetic_batch(distractors)
        for i in range(num_distractors):
            adapter.write_only(d_eos[i:i+1], d_is_eos[i:i+1], d_w_prob[i:i+1])

    # Read Target
    retrieved_v = adapter.read_only(t_eos)
    expected_v = adapter.get_v_proj(t_eos)

    sim = compute_cosine_similarity(retrieved_v, expected_v)
    return float(np.mean(sim))


def run_experiment(adapter=None, config=None, seeds=3):
    print("Running Interference Test (PyTorch)...")
    if adapter is None:
        adapter = MemoryBankAdapter()
        adapter.setup()
    dims = adapter.config.memory_dim
    sizes = [10, 50]
    noise_levels = [0.1, 0.5]
    results = {}
    for noise in noise_levels:
        for size in sizes:
            scores = []
            for seed in range(seeds):
                adapter.reset_memory()
                score = run_interference(adapter, size, dims, noise, seed=seed + 500)
                scores.append(score)
            results[(noise, size)] = {'mean': float(np.mean(scores)), 'std': float(np.std(scores))}
            print(f"  Noise {noise}, Distractors {size}: Sim = {results[(noise, size)]['mean']:.4f} ± {results[(noise, size)]['std']:.4f}")
    return results


if __name__ == '__main__':
    run_experiment()
