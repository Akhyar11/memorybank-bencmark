"""
experiments/counterfactual.py – Fixed True Causal Counterfactual Test.

Fix BUG-P1-003: Original used two completely different H_A and H_B.
True counterfactual requires:
  - Same key K (same h_eos for write)
  - Different stored VALUE (manipulate v_proj output)
  - Query with same K
  - Verify: retrieval changes only because value changed

Method:
  Experiment A: write K with params_A → v_proj_A(K) stored → read K → R_A
  Experiment B: same K with params_B (negated v_proj) → v_proj_B(K) stored → read K → R_B
  Assert: R_A ≠ R_B, sim(R_A, R_B) is low (causal effect of value)
"""
import jax
import jax.numpy as jnp
import flax.core
import numpy as np

from models.tiny_memory_bank import STATE_EXPIRED
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


def run_counterfactual(adapter, dim, key):
    """
    True Causal Counterfactual Test (Fix BUG-P1-003).

    Same key K → different stored values → different retrieval outputs.
    """
    key, k1 = jax.random.split(key)

    # Fixed query key K – used for both experiments
    K = jax.random.normal(k1, (1, dim))
    K = K / jnp.linalg.norm(K)

    # ---- Experiment A: original params ----
    adapter.reset_memory()
    adapter.write_only(K, jnp.ones((1,)), jnp.ones((1,)))
    R_A = adapter.read_only(K)

    # ---- Experiment B: negate v_proj → different stored value for same K ----
    # Manually flip v_proj kernel to produce V_B = -V_A
    unfrozen = flax.core.unfreeze(adapter.variables)
    unfrozen['params']['bank']['v_proj']['kernel'] = \
        -unfrozen['params']['bank']['v_proj']['kernel']
    adapter.variables = flax.core.freeze(unfrozen)

    adapter.reset_memory()
    adapter.write_only(K, jnp.ones((1,)), jnp.ones((1,)))
    R_B = adapter.read_only(K)

    # Restore original params
    unfrozen = flax.core.unfreeze(adapter.variables)
    unfrozen['params']['bank']['v_proj']['kernel'] = \
        -unfrozen['params']['bank']['v_proj']['kernel']
    adapter.variables = flax.core.freeze(unfrozen)

    # Expected A: v_proj_A(K)
    expected_A = adapter.get_v_proj(K)

    # Restore negated to get expected_B
    unfrozen = flax.core.unfreeze(adapter.variables)
    unfrozen['params']['bank']['v_proj']['kernel'] = \
        -unfrozen['params']['bank']['v_proj']['kernel']
    adapter.variables = flax.core.freeze(unfrozen)
    expected_B = adapter.get_v_proj(K)
    # Restore again
    unfrozen = flax.core.unfreeze(adapter.variables)
    unfrozen['params']['bank']['v_proj']['kernel'] = \
        -unfrozen['params']['bank']['v_proj']['kernel']
    adapter.variables = flax.core.freeze(unfrozen)

    sim_A     = float(jnp.mean(compute_cosine_similarity(R_A, expected_A)))
    sim_B     = float(jnp.mean(compute_cosine_similarity(R_B, expected_B)))
    sim_cross = float(jnp.mean(compute_cosine_similarity(R_A, R_B)))

    return sim_A, sim_B, sim_cross


def run_experiment(adapter, config, seeds=3):
    print("Running Counterfactual Causal Test (True Causal Intervention)...")
    dim = config.memory_dim

    scores_A, scores_B, scores_cross = [], [], []

    for seed in range(seeds):
        key = jax.random.PRNGKey(seed + 1000)
        sa, sb, sc = run_counterfactual(adapter, dim, key)
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
