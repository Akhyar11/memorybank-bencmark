"""
experiments/ablation.py – Fixed ablation experiment.

Fix BUG-P1-005: Original modified config AFTER JIT compile → had no effect.
Now each ablation creates a FRESH model with modified config.
This ensures the disabled mechanism is truly absent from computation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from models.tiny_memory_bank import TinyMemoryConfig, STATE_EXPIRED
from dataset.generator import generate_orthogonal_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity


def make_blank_memory(config):
    cap = config.memory_capacity
    dim = config.memory_dim
    return {
        'keys':         jnp.zeros((cap, dim),  dtype=jnp.float32),
        'vals':         jnp.zeros((cap, dim),  dtype=jnp.float32),
        'importance':   jnp.zeros((cap,),      dtype=jnp.float32),
        'confidence':   jnp.zeros((cap,),      dtype=jnp.float32),
        'created_at':   jnp.zeros((cap,),      dtype=jnp.int32),
        'last_access':  jnp.zeros((cap,),      dtype=jnp.int32),
        'access_count': jnp.zeros((cap,),      dtype=jnp.int32),
        'state':        jnp.full((cap,), STATE_EXPIRED, dtype=jnp.int32),
        'global_step':  jnp.zeros((),          dtype=jnp.int32),
    }


def make_ablation_config(ablation_type: str) -> TinyMemoryConfig:
    """
    Create a TinyMemoryConfig with the specified component disabled.
    Each ablation creates a NEW config (not modifying post-JIT).
    """
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
        base['mem_gamma'] = 0.0       # gamma = 0: recency not in score
    elif ablation_type == 'no_importance':
        base['mem_beta'] = 0.0        # beta = 0: importance not in score
    elif ablation_type == 'no_confidence':
        base['mem_delta'] = 0.0       # delta = 0: confidence not in score
    elif ablation_type == 'no_decay':
        base['mem_decay_rate'] = 0.0  # lambda = 0: no decay ever
    elif ablation_type == 'no_reinforcement':
        base['mem_reinforcement_rate'] = 0.0  # eta_a = 0: no importance boost on read
    else:
        raise ValueError(f"Unknown ablation_type: {ablation_type}")

    return TinyMemoryConfig(**base)


def run_ablation(ablation_type, num_memories, key, time_delay=100):
    """
    Run one ablation trial.
    Creates fresh model/vars for each trial to avoid JIT contamination.
    """
    from models.tiny_memory_bank import TinyMemoryBank

    config = make_ablation_config(ablation_type)
    bank   = TinyMemoryBank(config=config)
    rng    = jax.random.PRNGKey(0)
    h_init = jnp.ones((1, config.hidden_size))
    vars   = bank.init(rng, h_init, jnp.ones((1,)), jnp.ones((1,)), False)
    vars   = {'params': vars['params'], 'memory': make_blank_memory(config)}

    key, subkey = jax.random.split(key)
    dataset = generate_orthogonal_dataset(subkey, num_memories, config.memory_dim)

    # Write all memories
    for i in range(num_memories):
        h = dataset[i:i+1]
        _, new_mem = bank.apply(vars, h, jnp.ones((1,)), jnp.ones((1,)),
                                 method=bank.write, mutable=['memory'])
        vars = {'params': vars['params'], 'memory': new_mem['memory']}

    # Simulate time passing (for decay ablation comparison)
    if time_delay > 0:
        import flax.core
        unfrozen = flax.core.unfreeze(vars)
        unfrozen['memory']['global_step'] = jnp.array(time_delay, dtype=jnp.int32)
        vars = flax.core.freeze(unfrozen)
        _, new_mem = bank.apply(vars, method=bank.decay_memory, mutable=['memory'])
        vars = {'params': vars['params'], 'memory': new_mem['memory']}

    # Query target (middle element)
    target_idx    = num_memories // 2
    query_h       = dataset[target_idx:target_idx+1]
    read_val, _   = bank.apply(vars, query_h, method=bank.read, mutable=['memory'])
    expected_v, _ = bank.apply(vars, query_h, method=bank.v_proj, mutable=['memory'])

    sim = float(jnp.mean(compute_cosine_similarity(read_val, expected_v)))
    return sim


def run_experiment(config_path, config, seeds=3):
    print("Running Ablation Test (Fresh Model Per Ablation)...")
    num_memories = 50
    ablations    = [
        'none',
        'no_recency',
        'no_importance',
        'no_confidence',
        'no_decay',
        'no_reinforcement',
    ]
    results = {}

    for ab in ablations:
        scores = []
        for seed in range(seeds):
            key   = jax.random.PRNGKey(seed + 2000)
            score = run_ablation(ab, num_memories, key, time_delay=100)
            scores.append(score)
        results[ab] = {'mean': np.mean(scores), 'std': np.std(scores)}
        print(f"  Ablation '{ab:20s}': Sim = {results[ab]['mean']:.4f} ± {results[ab]['std']:.4f}")

    return results
