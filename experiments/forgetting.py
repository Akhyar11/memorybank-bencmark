"""
experiments/forgetting.py – Forgetting / Decay Experiment (PyTorch Version).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from dataset.generator import generate_random_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity
from adapters.existing_memorybank import MemoryBankAdapter


def run_forgetting(adapter, dim, time_steps, seed):
    """
    Forgetting / Decay Test.
    Write target. Simulate time steps. Observe if memory becomes dormant/expired.
    """
    target = generate_random_dataset(seed, 1, dim)
    t_eos, t_is_eos, t_w_prob, _ = create_synthetic_batch(target)

    # Write Target
    adapter.reset_memory()
    adapter.write_only(t_eos, t_is_eos, t_w_prob)

    # Initial read
    retrieved_v_initial = adapter.read_only(t_eos)
    expected_v = adapter.get_v_proj(t_eos)
    sim_initial = compute_cosine_similarity(retrieved_v_initial, expected_v)

    # Advance time
    adapter.advance_time(time_steps)
    adapter.decay_memory()

    # Read again
    retrieved_v_final = adapter.read_only(t_eos)
    sim_final = compute_cosine_similarity(retrieved_v_final, expected_v)

    state = adapter.get_memory_state()[0][0]
    return float(np.mean(sim_initial)), float(np.mean(sim_final)), int(state)


def run_experiment(adapter=None, config=None, seeds=3):
    print("Running Forgetting Test (PyTorch)...")
    if adapter is None:
        adapter = MemoryBankAdapter()
        adapter.setup()
    dims = adapter.config.memory_dim
    times = [0, 10, 1000, 10000]
    results = {}

    for t in times:
        scores_initial = []
        scores_final = []
        states = []
        for seed in range(seeds):
            si, sf, st = run_forgetting(adapter, dims, t, seed=seed)
            scores_initial.append(si)
            scores_final.append(sf)
            states.append(st)

        results[t] = {
            'initial': float(np.mean(scores_initial)),
            'final': float(np.mean(scores_final)),
            'state_mode': max(set(states), key=states.count)
        }
        print(f"  T={t:5d}: Initial Sim={results[t]['initial']:.4f}, Final Sim={results[t]['final']:.4f}, State={results[t]['state_mode']}")

    return results


if __name__ == '__main__':
    run_experiment()
