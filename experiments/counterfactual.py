"""
experiments/counterfactual.py – True Causal Counterfactual Test (PyTorch Version).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from models.tiny_memory_bank import STATE_EXPIRED
from evaluation.metrics import compute_cosine_similarity
from adapters.existing_memorybank import MemoryBankAdapter


def run_counterfactual(adapter, dim, seed):
    """
    True Causal Counterfactual Test.
    Same key K → different stored values → different retrieval outputs.
    """
    torch.manual_seed(seed)
    K = torch.randn(1, dim)
    K = (K / torch.norm(K)).numpy()

    # ---- Experiment A: original params ----
    adapter.reset_memory()
    adapter.write_only(K, np.ones(1), np.ones(1))
    R_A = adapter.read_only(K)
    expected_A = adapter.get_v_proj(K)

    # ---- Experiment B: negate v_proj → different stored value for same K ----
    with torch.no_grad():
        adapter.model.bank.v_proj.weight.data.neg_()

    adapter.reset_memory()
    adapter.write_only(K, np.ones(1), np.ones(1))
    R_B = adapter.read_only(K)
    expected_B = adapter.get_v_proj(K)

    # Restore original params
    with torch.no_grad():
        adapter.model.bank.v_proj.weight.data.neg_()

    sim_A     = float(np.mean(compute_cosine_similarity(R_A, expected_A)))
    sim_B     = float(np.mean(compute_cosine_similarity(R_B, expected_B)))
    sim_cross = float(np.mean(compute_cosine_similarity(R_A, R_B)))

    return sim_A, sim_B, sim_cross


def run_experiment(adapter=None, config=None, seeds=3):
    print("Running Counterfactual Causal Test (PyTorch)...")
    if adapter is None:
        adapter = MemoryBankAdapter()
        adapter.setup()
    dim = adapter.config.memory_dim

    scores_A, scores_B, scores_cross = [], [], []

    for seed in range(seeds):
        sa, sb, sc = run_counterfactual(adapter, dim, seed=seed + 1000)
        scores_A.append(sa)
        scores_B.append(sb)
        scores_cross.append(sc)

    results = {
        'A':     {'mean': np.mean(scores_A),     'std': np.std(scores_A)},
        'B':     {'mean': np.mean(scores_B),     'std': np.std(scores_B)},
        'cross': {'mean': np.mean(scores_cross), 'std': np.std(scores_cross)},
    }

    print(f"  Match A (RA vs expected_A):  {results['A']['mean']:.4f} ± {results['A']['std']:.4f}")
    print(f"  Match B (RB vs expected_B):  {results['B']['mean']:.4f} ± {results['B']['std']:.4f}")
    print(f"  Cross (RA vs RB, should differ): {results['cross']['mean']:.4f} ± {results['cross']['std']:.4f}")
    print(f"  Causal effect: {'YES' if results['cross']['mean'] < 0.9 else 'NO'}")
    return results


if __name__ == '__main__':
    run_experiment()
