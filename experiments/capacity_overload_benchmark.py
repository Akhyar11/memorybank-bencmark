"""
experiments/capacity_overload_benchmark.py – Capacity & Overload Benchmark (P1).

Measures retention rate, eviction rate, false retrieval rate, Recall@1, Recall@5, and MRR
across under-capacity, at-capacity, and over-capacity conditions.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import collections

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig
from evaluation.metrics import recall_at_k, mean_reciprocal_rank


def run_capacity_test(capacity: int, num_facts: int, dim: int = 16, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    config = TinyMemoryConfig(
        memory_capacity=capacity,
        memory_dim=dim,
        hidden_size=dim,
        memory_top_k=4,
        memory_write_threshold=0.9
    )
    bank = TinyMemoryBank(config=config)
    bank.load_memory_state(bank.empty_memory_state())

    facts = torch.randn(num_facts, dim, generator=g)
    facts = facts / torch.norm(facts, dim=-1, keepdim=True)

    fact_to_slot = {}
    slot_to_fact = {}

    # Write all facts sequentially
    for i in range(num_facts):
        h = facts[i:i+1]
        target_idx = bank.write(h, torch.ones(1), torch.ones(1))[0].item()
        f_id = f"fact_{i}"
        if target_idx in slot_to_fact:
            evicted_fact = slot_to_fact[target_idx]
            if evicted_fact in fact_to_slot:
                del fact_to_slot[evicted_fact]
        slot_to_fact[target_idx] = f_id
        fact_to_slot[f_id] = target_idx

    # Query all written facts
    retained_facts = 0
    false_retrievals = 0
    recalls_1 = []
    recalls_5 = []
    mrrs = []

    for i in range(num_facts):
        f_id = f"fact_{i}"
        h = facts[i:i+1]
        _, scores = bank.read(h, return_scores=True)
        scores_np = scores[0].detach().cpu().numpy()

        is_retained = f_id in fact_to_slot
        if is_retained:
            retained_facts += 1
            gt_slot = fact_to_slot[f_id]
            r = recall_at_k(scores_np, gt_slot, k_values=[1, 5])
            mrr = mean_reciprocal_rank(scores_np, gt_slot)
            recalls_1.append(r[1])
            recalls_5.append(r[5])
            mrrs.append(mrr)
        else:
            # Fact was evicted: evaluate false retrieval
            # Top-1 retrieved slot
            top1_slot = int(np.argmax(scores_np))
            if scores_np[top1_slot] > config.memory_threshold:
                false_retrievals += 1
            recalls_1.append(0.0)
            recalls_5.append(0.0)
            mrrs.append(0.0)

    retention_rate = retained_facts / num_facts
    eviction_rate = 1.0 - retention_rate
    false_retrieval_rate = false_retrievals / max(num_facts - retained_facts, 1) if num_facts > retained_facts else 0.0

    return {
        'capacity': capacity,
        'num_facts': num_facts,
        'retention_rate': retention_rate,
        'eviction_rate': eviction_rate,
        'false_retrieval_rate': false_retrieval_rate,
        'recall@1': float(np.mean(recalls_1)),
        'recall@5': float(np.mean(recalls_5)),
        'mrr': float(np.mean(mrrs))
    }


def run_benchmark(capacities=(16, 32, 64), fact_multipliers=(0.5, 1.0, 2.0, 4.0), seeds=(42, 43, 44)):
    print("=" * 75)
    print("        MEMORY CAPACITY & OVERLOAD BENCHMARK (P1 Sweep)")
    print("=" * 75)
    print(f"{'Cap':<6s} {'Facts':<8s} {'Condition':<15s} {'Retention':<12s} {'Eviction':<12s} {'Recall@1':<12s} {'MRR':<10s}")
    print("-" * 75)

    results = collections.defaultdict(list)

    for cap in capacities:
        for mult in fact_multipliers:
            n_facts = max(int(cap * mult), 4)
            if mult < 1.0:
                cond = "Under-Cap"
            elif mult == 1.0:
                cond = "At-Capacity"
            else:
                cond = "Overload"

            runs = [run_capacity_test(cap, n_facts, seed=s) for s in seeds]
            avg_ret = np.mean([r['retention_rate'] for r in runs])
            avg_evict = np.mean([r['eviction_rate'] for r in runs])
            avg_r1 = np.mean([r['recall@1'] for r in runs])
            avg_mrr = np.mean([r['mrr'] for r in runs])

            print(f"{cap:<6d} {n_facts:<8d} {cond:<15s} {avg_ret*100:>5.1f}%       {avg_evict*100:>5.1f}%       {avg_r1:>5.3f}        {avg_mrr:>5.3f}")

    print("=" * 75)


if __name__ == '__main__':
    run_benchmark()
