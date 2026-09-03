"""
experiments/long_term_memory_benchmark.py – True Episodic Long-Term Memory Benchmark.

Features:
- Multi-turn episodes where Query is separated from Write by configurable delay & distractors.
- Fact-ID to physical slot explicit resolution (robust to eviction/replacement).
- Actual top-K ranking and composite score extraction from Memory Bank.
- Evaluates Recall@1, Recall@5, MRR, and Rank across delays [0, 8, 32, 128] and multiple seeds.
- Compares Memory Bank against Independent Key-Value NN Memory and No-Memory.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import collections

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig
from baselines.nearest_neighbor import NearestNeighborMemory
from baselines.no_memory import NoMemory
from evaluation.metrics import recall_at_k, mean_reciprocal_rank


def run_long_term_episode(
    bank_or_baseline,
    mode: str,
    dim: int,
    capacity: int,
    num_facts: int = 10,
    delay: int = 32,
    seed: int = 42,
    time_progression: bool = True
):
    """
    Runs a single long-term memory episode:
    1. Writes N target facts with logical IDs ('fact_0', 'fact_1', ...)
    2. Tracks physical slot mapping: fact_to_slot and slot_to_fact
    3. Writes D distractor facts ('distractor_0', ...)
    4. Advances temporal clock if time_progression is True
    5. Queries target facts that were written far in the past (delay steps ago)
    6. Evaluates actual retrieval metrics based on logical fact identity
    """
    g = torch.Generator().manual_seed(seed)

    # Tracking structures
    slot_to_fact = {}
    fact_to_slot = {}

    # 1. Generate target facts
    target_embs = torch.randn(num_facts, dim, generator=g)
    target_embs = target_embs / torch.norm(target_embs, dim=-1, keepdim=True)
    target_facts = [f"fact_{i}" for i in range(num_facts)]

    # Write target facts
    for i, f_id in enumerate(target_facts):
        h = target_embs[i:i+1]
        if mode == 'bank':
            written_slot = bank_or_baseline.write(h, torch.ones(1), torch.ones(1))[0].item()
        elif mode == 'nn':
            slot = i % capacity
            bank_or_baseline.write(h, h, slot_idx=slot)
            written_slot = slot
        else:
            written_slot = -1

        if written_slot >= 0:
            # If slot previously held another fact, evict it from fact_to_slot
            if written_slot in slot_to_fact:
                evicted = slot_to_fact[written_slot]
                if evicted in fact_to_slot and fact_to_slot[evicted] == written_slot:
                    del fact_to_slot[evicted]
            slot_to_fact[written_slot] = f_id
            fact_to_slot[f_id] = written_slot

    # 2. Write distractors (introducing delay)
    if delay > 0:
        distractor_embs = torch.randn(delay, dim, generator=g)
        distractor_embs = distractor_embs / torch.norm(distractor_embs, dim=-1, keepdim=True)

        for d in range(delay):
            h_d = distractor_embs[d:d+1]
            f_distractor = f"distractor_{d}"
            if mode == 'bank':
                if time_progression:
                    bank_or_baseline.global_step[0] += 1
                written_slot = bank_or_baseline.write(h_d, torch.ones(1), torch.ones(1))[0].item()
            elif mode == 'nn':
                slot = (num_facts + d) % capacity
                bank_or_baseline.write(h_d, h_d, slot_idx=slot)
                written_slot = slot
            else:
                written_slot = -1

            if written_slot >= 0:
                if written_slot in slot_to_fact:
                    evicted = slot_to_fact[written_slot]
                    if evicted in fact_to_slot and fact_to_slot[evicted] == written_slot:
                        del fact_to_slot[evicted]
                slot_to_fact[written_slot] = f_distractor
                fact_to_slot[f_distractor] = written_slot

    if mode == 'bank' and time_progression:
        bank_or_baseline.decay_memory()

    # 3. Query the earliest written target fact (e.g. fact_0, max delay from present)
    recalls_1 = []
    recalls_5 = []
    mrrs = []
    ranks = []

    for i in range(min(num_facts, 5)):
        query_fact = target_facts[i]
        query_emb = target_embs[i:i+1]

        # Ground truth slot for this fact
        gt_slot = fact_to_slot.get(query_fact, -1)

        if mode == 'bank':
            with torch.no_grad():
                _, scores = bank_or_baseline.read(query_emb, return_scores=True)
            scores_np = scores[0].detach().cpu().numpy()
        elif mode == 'nn':
            with torch.no_grad():
                _, sim_scores, _ = bank_or_baseline.read(query_emb, return_scores=True)
            scores_np = sim_scores[0].detach().cpu().numpy()
        else:
            # No-memory produces arbitrary zero retrieval
            scores_np = np.zeros(capacity)

        if gt_slot >= 0 and gt_slot < len(scores_np):
            r = recall_at_k(scores_np, gt_slot, k_values=[1, 5])
            mrr = mean_reciprocal_rank(scores_np, gt_slot)
            # Find actual rank
            sorted_indices = np.argsort(scores_np)[::-1]
            rank_matches = np.where(sorted_indices == gt_slot)[0]
            rank = int(rank_matches[0] + 1) if len(rank_matches) > 0 else -1
        else:
            # Fact was evicted due to capacity constraints or no memory
            r = {1: 0.0, 5: 0.0}
            mrr = 0.0
            rank = -1

        recalls_1.append(r[1])
        recalls_5.append(r[5])
        mrrs.append(mrr)
        ranks.append(rank)

    return {
        'recall@1': float(np.mean(recalls_1)),
        'recall@5': float(np.mean(recalls_5)),
        'mrr': float(np.mean(mrrs)),
        'ranks': ranks
    }


def run_benchmark(
    delays=(0, 8, 32, 128),
    capacity=64,
    dim=32,
    seeds=(42, 43, 44),
    num_facts=10
):
    print("=" * 70)
    print("     EPISODIC LONG-TERM MEMORY BENCHMARK (Multi-Delay & Multi-Seed)")
    print("=" * 70)
    print(f"Capacity: {capacity} | Dim: {dim} | Facts: {num_facts} | Seeds: {list(seeds)}")
    print(f"Tested Delays: {list(delays)}")
    print("=" * 70)

    all_results = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))

    modes = ['no_memory', 'nn', 'bank']

    for delay in delays:
        print(f"\n[EVALUATING DELAY = {delay:3d} distractors]")

        for mode in modes:
            for seed in seeds:
                if mode == 'bank':
                    cfg = TinyMemoryConfig(memory_capacity=capacity, memory_dim=dim, hidden_size=dim, mem_decay_rate=0.001)
                    torch.manual_seed(seed)
                    model = TinyMemoryBank(config=cfg)
                    model.load_memory_state(model.empty_memory_state())
                elif mode == 'nn':
                    torch.manual_seed(seed)
                    model = NearestNeighborMemory(capacity=capacity, dim=dim)
                else:
                    model = NoMemory(dim=dim)

                res = run_long_term_episode(
                    model, mode=mode, dim=dim, capacity=capacity,
                    num_facts=num_facts, delay=delay, seed=seed
                )

                all_results[delay][mode]['recall@1'].append(res['recall@1'])
                all_results[delay][mode]['recall@5'].append(res['recall@5'])
                all_results[delay][mode]['mrr'].append(res['mrr'])

            m_r1 = np.mean(all_results[delay][mode]['recall@1'])
            s_r1 = np.std(all_results[delay][mode]['recall@1'])
            m_r5 = np.mean(all_results[delay][mode]['recall@5'])
            s_r5 = np.std(all_results[delay][mode]['recall@5'])
            m_mrr = np.mean(all_results[delay][mode]['mrr'])
            s_mrr = np.std(all_results[delay][mode]['mrr'])

            print(f"  {mode:<12s} | R@1: {m_r1:.3f}±{s_r1:.3f} | R@5: {m_r5:.3f}±{s_r5:.3f} | MRR: {m_mrr:.3f}±{s_mrr:.3f}")

    print("\n" + "=" * 70)
    print("     LONG-TERM BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Delay':<8s} {'Mode':<12s} {'Recall@1':<18s} {'Recall@5':<18s} {'MRR':<18s}")
    print("-" * 70)
    for delay in delays:
        for mode in modes:
            r1 = f"{np.mean(all_results[delay][mode]['recall@1']):.3f}±{np.std(all_results[delay][mode]['recall@1']):.3f}"
            r5 = f"{np.mean(all_results[delay][mode]['recall@5']):.3f}±{np.std(all_results[delay][mode]['recall@5']):.3f}"
            mrr = f"{np.mean(all_results[delay][mode]['mrr']):.3f}±{np.std(all_results[delay][mode]['mrr']):.3f}"
            print(f"{delay:<8d} {mode:<12s} {r1:<18s} {r5:<18s} {mrr:<18s}")
    print("=" * 70)
    return all_results


if __name__ == '__main__':
    run_benchmark()
