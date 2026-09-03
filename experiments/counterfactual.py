"""
experiments/counterfactual.py – Advanced Controlled Counterfactual & Perturbation Test (P1).

Tests:
1. Minimally Perturbed Counterfactual: Fact A vs Fact A' (high cosine similarity ~0.90-0.95)
   Evaluates whether Memory Bank correctly resolves the exact target fact vs the perturbed variant.
   Reports Recall@1, Recall@K, and MRR.
2. Causal Parameter Intervention: Negation of projection weights proves true causality.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig
from evaluation.metrics import compute_cosine_similarity, recall_at_k, mean_reciprocal_rank
from adapters.existing_memorybank import MemoryBankAdapter


def run_perturbed_counterfactual(dim: int = 32, perturbation_eps: float = 0.15, seed: int = 42):
    """
    Minimally perturbed counterfactual test:
    Fact A and Fact A' (A' = normalize(A + eps * noise)).
    Both stored in Memory Bank along with distractors.
    Query A must retrieve slot A, NOT slot A'.
    """
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    config = TinyMemoryConfig(
        memory_capacity=16, memory_dim=dim, hidden_size=dim,
        memory_top_k=4, memory_write_threshold=0.9
    )
    bank = TinyMemoryBank(config=config)
    bank.load_memory_state(bank.empty_memory_state())

    # Generate base Fact A
    fact_A = torch.randn(1, dim, generator=g)
    fact_A = fact_A / torch.norm(fact_A)

    # Generate minimally perturbed Fact A'
    noise = torch.randn(1, dim, generator=g)
    fact_A_prime = fact_A + perturbation_eps * noise
    fact_A_prime = fact_A_prime / torch.norm(fact_A_prime)

    perturbation_similarity = float(torch.sum(fact_A * fact_A_prime))

    # Write Fact A (slot 0)
    slot_A = bank.write(fact_A, torch.ones(1), torch.ones(1))[0].item()

    # Write Fact A' (slot 1)
    slot_A_prime = bank.write(fact_A_prime, torch.ones(1), torch.ones(1))[0].item()

    # Write several unrelated distractors
    for _ in range(6):
        d = torch.randn(1, dim, generator=g)
        d = d / torch.norm(d)
        bank.write(d, torch.ones(1), torch.ones(1))

    # Query with Fact A
    with torch.no_grad():
        _, scores = bank.read(fact_A, return_scores=True)
    scores_np = scores[0].detach().cpu().numpy()

    # Evaluate retrieval of Fact A
    r = recall_at_k(scores_np, slot_A, k_values=[1, 2, 5])
    mrr = mean_reciprocal_rank(scores_np, slot_A)

    top_slot = int(np.argmax(scores_np))
    correct = (top_slot == slot_A)

    return {
        'perturbation_similarity': perturbation_similarity,
        'slot_A': slot_A,
        'slot_A_prime': slot_A_prime,
        'top_retrieved_slot': top_slot,
        'correct': correct,
        'recall@1': r[1],
        'recall@2': r[2],
        'recall@5': r[5],
        'mrr': mrr
    }


def run_causal_v_proj_negation(adapter, dim, seed):
    """Causal value intervention test (negated projection weights)."""
    torch.manual_seed(seed)
    K = torch.randn(1, dim)
    K = (K / torch.norm(K)).numpy()

    adapter.reset_memory()
    adapter.write_only(K, np.ones(1), np.ones(1))
    R_A = adapter.read_only(K)
    expected_A = adapter.get_v_proj(K)

    with torch.no_grad():
        adapter.model.bank.v_proj.weight.data.neg_()

    adapter.reset_memory()
    adapter.write_only(K, np.ones(1), np.ones(1))
    R_B = adapter.read_only(K)
    expected_B = adapter.get_v_proj(K)

    # Restore
    with torch.no_grad():
        adapter.model.bank.v_proj.weight.data.neg_()

    sim_A = float(np.mean(compute_cosine_similarity(R_A, expected_A)))
    sim_B = float(np.mean(compute_cosine_similarity(R_B, expected_B)))
    sim_cross = float(np.mean(compute_cosine_similarity(R_A, R_B)))

    return sim_A, sim_B, sim_cross


def run_experiment(adapter=None, config=None, seeds=3):
    print("Running Controlled Counterfactual & Perturbation Tests (PyTorch)...")
    if adapter is None:
        adapter = MemoryBankAdapter()
        adapter.setup()
    dim = adapter.config.memory_dim

    print("\n1. Minimally Perturbed Facts Test (Fact A vs Fact A'):")
    perturbed_runs = [run_perturbed_counterfactual(dim=dim, seed=s + 100) for s in range(seeds)]
    avg_sim = np.mean([r['perturbation_similarity'] for r in perturbed_runs])
    avg_r1 = np.mean([r['recall@1'] for r in perturbed_runs])
    avg_r2 = np.mean([r['recall@2'] for r in perturbed_runs])
    avg_mrr = np.mean([r['mrr'] for r in perturbed_runs])

    print(f"  Perturbation Cosine Similarity: {avg_sim:.4f}")
    print(f"  Recall@1: {avg_r1:.4f} | Recall@2: {avg_r2:.4f} | MRR: {avg_mrr:.4f}")

    print("\n2. Causal Projection Intervention Test (v_proj negation):")
    causal_runs = [run_causal_v_proj_negation(adapter, dim, seed=s) for s in range(seeds)]
    avg_cross = np.mean([c[2] for c in causal_runs])
    print(f"  Cross-Similarity (R_A vs R_B): {avg_cross:.4f} (Ideal: -1.0000)")

    return {
        'perturbed_r1': avg_r1,
        'perturbed_mrr': avg_mrr,
        'causal_cross_sim': avg_cross
    }


if __name__ == '__main__':
    run_experiment()
