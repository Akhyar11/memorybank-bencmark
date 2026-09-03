"""
experiments/ablation.py (PyTorch Version)

Ablation experiment for TinyMemoryBank components.
Each ablation creates a fresh model with modified config to ensure the disabled
mechanism is truly absent from computation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, STATE_EXPIRED, STATE_ACTIVE
from dataset.generator import generate_orthogonal_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity


def make_ablation_config(ablation_type: str) -> TinyMemoryConfig:
    """Create a TinyMemoryConfig with the specified component disabled."""
    base = dict(
        memory_capacity=64,
        memory_dim=16,
        hidden_size=16,
        memory_top_k=8,
        memory_threshold=0.0,
        memory_write_threshold=0.0,
        mem_decay_rate=0.01,
        mem_importance_protection=0.5,
        mem_alpha=1.0,
        mem_beta=0.5,
        mem_gamma=0.1,
        mem_delta=0.2,
        mem_reinforcement_rate=0.05,
    )

    if ablation_type == 'none':
        pass  # full model
    elif ablation_type == 'no_recency':
        base['mem_gamma'] = 0.0
    elif ablation_type == 'no_importance':
        base['mem_beta'] = 0.0
    elif ablation_type == 'no_confidence':
        base['mem_delta'] = 0.0
    elif ablation_type == 'no_decay':
        base['mem_decay_rate'] = 0.0
    elif ablation_type == 'no_reinforcement':
        base['mem_reinforcement_rate'] = 0.0
    elif ablation_type == 'no_top_k':
        base['memory_top_k'] = base['memory_capacity']
    elif ablation_type == 'no_retrieval_threshold':
        base['memory_threshold'] = -1e9
    elif ablation_type == 'no_write_gate':
        base['memory_write_threshold'] = -1e9
    elif ablation_type == 'no_read_gate':
        base['memory_read_threshold'] = -1e9
    elif ablation_type == 'no_write':
        base['memory_write_threshold'] = 1e9
    elif ablation_type == 'no_read':
        base['memory_read_threshold'] = 1e9
    else:
        raise ValueError(f"Unknown ablation_type: {ablation_type}")

    return TinyMemoryConfig(**base)


def run_ablation(ablation_type, num_memories, seed=0, time_delay=100):
    """Run one ablation trial."""
    config = make_ablation_config(ablation_type)
    torch.manual_seed(seed)
    bank = TinyMemoryBank(config=config)
    bank.load_memory_state(bank.empty_memory_state())

    dataset = generate_orthogonal_dataset(seed, num_memories, config.memory_dim)

    # Write all memories
    mem_before_write = bank.get_memory_state()
    for i in range(num_memories):
        h = dataset[i:i+1]
        bank.write(h, torch.ones(1), torch.ones(1))

    if ablation_type == 'no_write':
        mem_after_write = bank.get_memory_state()
        for k in ['state', 'keys', 'vals', 'created_at', 'importance', 'confidence']:
            assert torch.equal(mem_before_write[k], mem_after_write[k]), f"no_write altered {k}"

    # Simulate time passing
    if time_delay > 0:
        bank.global_step[0] = time_delay
        bank.decay_memory()

    # Query target (middle element)
    target_idx = num_memories // 2
    query_h = dataset[target_idx:target_idx+1]
    read_prob_val = torch.ones(1)

    mem_before_read = bank.get_memory_state()
    read_val = bank.read(query_h, read_prob=read_prob_val)
    mem_after_read = bank.get_memory_state()

    expected_v = bank.v_proj(query_h)

    if ablation_type == 'no_write':
        active_count = int(torch.sum(bank.mem_state == STATE_ACTIVE))
        assert active_count == 0, f"no_write failed to block writes! Active memory: {active_count}"
        assert float(torch.norm(read_val)) == 0.0, "no_write read result is not exactly 0.0"

    if ablation_type == 'no_read':
        assert float(torch.norm(read_val)) == 0.0, "no_read read result is not exactly 0.0"
        for k in ['last_access', 'access_count', 'importance']:
            assert torch.equal(mem_before_read[k], mem_after_read[k]), f"no_read altered {k}"

    sim = float(np.mean(compute_cosine_similarity(read_val.detach().numpy(), expected_v.detach().numpy())))
    return sim


def run_experiment(seeds=3):
    print("Running Ablation Test (Fresh Model Per Ablation - PyTorch)...")
    num_memories = 50
    ablations = [
        'none',
        'no_recency',
        'no_importance',
        'no_confidence',
        'no_decay',
        'no_reinforcement',
        'no_top_k',
        'no_retrieval_threshold',
        'no_write_gate',
        'no_read_gate',
        'no_write',
        'no_read',
    ]
    results = {}

    for ab in ablations:
        scores = []
        for seed in range(seeds):
            score = run_ablation(ab, num_memories, seed=seed + 2000, time_delay=100)
            scores.append(score)
        results[ab] = {'mean': float(np.mean(scores)), 'std': float(np.std(scores))}
        print(f"  Ablation '{ab:20s}': Sim = {results[ab]['mean']:.4f} ± {results[ab]['std']:.4f}")

    return results


if __name__ == '__main__':
    run_experiment()
