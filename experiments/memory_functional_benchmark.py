"""
experiments/memory_functional_benchmark.py – STEP 9 / Requirement #27

Complete 11-test Memory Bank functional benchmark.
Tests the Memory Bank itself independently of the language model.

Tests:
  1.  Basic Write
  2.  Basic Read
  3.  Distractor Retrieval (Recall@1, Recall@K, MRR)
  4.  Interference
  5.  Capacity Scaling
  6.  Replacement Policy
  7.  Recency Effect
  8.  Importance Effect
  9.  Confidence Effect
  10. Forgetting (ACTIVE → DORMANT → EXPIRED)
  11. Counterfactual Causal Test
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import flax.core
import numpy as np

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, STATE_ACTIVE, STATE_EXPIRED, STATE_DORMANT
from evaluation.metrics import compute_cosine_similarity, recall_at_k, mean_reciprocal_rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_config(capacity=32, dim=16, hidden=16, top_k=4,
                threshold=0.0, write_threshold=0.0,
                decay_rate=0.0001):
    return TinyMemoryConfig(
        memory_capacity=capacity, memory_dim=dim, hidden_size=hidden,
        memory_top_k=top_k, memory_threshold=threshold,
        memory_write_threshold=write_threshold,
        mem_decay_rate=decay_rate,
        mem_importance_protection=0.5,
        mem_alpha=1.0, mem_beta=0.5, mem_gamma=0.1, mem_delta=0.2,
        mem_reinforcement_rate=0.05,
    )


def fresh_bank(config, seed=0):
    """Create bank with fresh (empty) memory."""
    bank = TinyMemoryBank(config=config)
    rng  = jax.random.PRNGKey(seed)
    h0   = jnp.ones((1, config.hidden_size))
    vars = bank.init(rng, h0, jnp.ones((1,)), jnp.ones((1,)), False)
    # Overwrite memory with truly empty state
    mem = {
        'keys':         jnp.zeros((config.memory_capacity, config.memory_dim)),
        'vals':         jnp.zeros((config.memory_capacity, config.memory_dim)),
        'importance':   jnp.zeros((config.memory_capacity,)),
        'confidence':   jnp.zeros((config.memory_capacity,)),
        'created_at':   jnp.zeros((config.memory_capacity,), jnp.int32),
        'last_access':  jnp.zeros((config.memory_capacity,), jnp.int32),
        'access_count': jnp.zeros((config.memory_capacity,), jnp.int32),
        'state':        jnp.full((config.memory_capacity,), STATE_EXPIRED, jnp.int32),
        'global_step':  jnp.zeros((), jnp.int32),
    }
    vars = {'params': vars['params'], 'memory': mem}
    return bank, vars


def write_one(bank, vars, h, eos=None, wp=None):
    if eos is None: eos = jnp.ones((h.shape[0],))
    if wp  is None: wp  = jnp.ones((h.shape[0],))
    _, new_mem = bank.apply(vars, h, eos, wp, method=bank.write, mutable=['memory'])
    return {'params': vars['params'], 'memory': new_mem['memory']}


def read_one(bank, vars, h):
    out, new_mem = bank.apply(vars, h, method=bank.read, mutable=['memory'])
    vars = {'params': vars['params'], 'memory': new_mem['memory']}
    return out, vars


def cosine_float(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return 0.0
    return float(np.dot(a/na, b/nb))


# ---------------------------------------------------------------------------
# Test 1: Basic Write
# ---------------------------------------------------------------------------
def test1_basic_write():
    print("\n[TEST 1] Basic Write")
    config = make_config()
    bank, vars = fresh_bank(config, seed=1)

    h = jax.random.normal(jax.random.PRNGKey(1), (1, config.hidden_size))
    vars = write_one(bank, vars, h)

    active = int(jnp.sum(vars['memory']['state'] == STATE_ACTIVE))
    result = {"active_after_write": active, "expected": 1, "pass": active == 1}
    print(f"  active_count after 1 write = {active}  → {'PASS' if result['pass'] else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 2: Basic Read
# ---------------------------------------------------------------------------
def test2_basic_read():
    print("\n[TEST 2] Basic Read")
    config = make_config()
    bank, vars = fresh_bank(config, seed=2)

    h = jax.random.normal(jax.random.PRNGKey(2), (1, config.hidden_size))
    h = h / jnp.linalg.norm(h)

    vars        = write_one(bank, vars, h)
    R, _        = read_one(bank, vars, h)
    # Also apply the actual `v_proj` using the current parameters.
    expected_v, _ = bank.apply(vars, h, method=lambda mdl, x: mdl.v_proj(x), mutable=['memory'])

    sim = float(jnp.sum(R[0] * expected_v[0]) / (jnp.linalg.norm(R[0]) * jnp.linalg.norm(expected_v[0]) + 1e-8))
    passed = sim > 0.1  # generous: bank randomly initialised
    result = {"cosine_sim": sim, "pass": passed}
    print(f"  Write-Read cosine similarity = {sim:.4f}  → {'PASS' if passed else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 3: Distractor Retrieval
# ---------------------------------------------------------------------------
def test3_distractor_retrieval(n_distractors=10, seeds=3):
    print(f"\n[TEST 3] Distractor Retrieval (N={n_distractors})")
    config = make_config(capacity=64, top_k=8)

    r1_scores, mrr_scores = [], []
    for seed in range(seeds):
        bank, vars = fresh_bank(config, seed=seed)
        rng = jax.random.PRNGKey(seed + 100)

        rng, k1 = jax.random.split(rng)
        h_target = jax.random.normal(k1, (1, config.hidden_size))
        h_target = h_target / jnp.linalg.norm(h_target)
        vars = write_one(bank, vars, h_target)
        gt_idx = 0  # first slot written

        for i in range(n_distractors):
            rng, k = jax.random.split(rng)
            h_d = jax.random.normal(k, (1, config.hidden_size))
            h_d = h_d / jnp.linalg.norm(h_d)
            vars = write_one(bank, vars, h_d)

        # Compute per-slot cosine scores against query
        q, _ = bank.apply(vars, h_target, method=lambda mdl, x: mdl.q_proj(x), mutable=['memory'])
        keys  = vars['memory']['keys']
        q_n   = q[0] / (jnp.linalg.norm(q[0]) + 1e-8)
        k_n   = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        scores = np.array(jnp.matmul(q_n, k_n.T))

        state  = np.array(vars['memory']['state'])
        # Mask inactive
        scores[state != STATE_ACTIVE] = -1e9

        r1  = recall_at_k(scores, gt_idx, k_values=[1, config.memory_top_k])
        mrr = mean_reciprocal_rank(scores, gt_idx)
        r1_scores.append(r1[1])
        mrr_scores.append(mrr)

    result = {
        "Recall@1":   np.mean(r1_scores),
        "MRR":        np.mean(mrr_scores),
        "pass": np.mean(r1_scores) >= 0.5,  # Needs to actually recall correctly at least half the time
    }
    print(f"  Recall@1={result['Recall@1']:.4f}  MRR={result['MRR']:.4f}")
    return result


# ---------------------------------------------------------------------------
# Test 4: Interference
# ---------------------------------------------------------------------------
def test4_interference(n_distractors=20, seeds=3):
    print(f"\n[TEST 4] Interference (N={n_distractors})")
    config = make_config(capacity=64, top_k=8)

    sims = []
    for seed in range(seeds):
        bank, vars = fresh_bank(config, seed=seed)
        rng = jax.random.PRNGKey(seed + 200)

        rng, k1 = jax.random.split(rng)
        h_target = jax.random.normal(k1, (1, config.hidden_size))
        h_target = h_target / jnp.linalg.norm(h_target)
        vars = write_one(bank, vars, h_target)
        vars_clean = {'params': vars['params'], 'memory': {k: v for k, v in vars['memory'].items()}}

        for i in range(n_distractors):
            rng, k = jax.random.split(rng)
            h_d = jax.random.normal(k, (1, config.hidden_size))
            vars = write_one(bank, vars, h_d)

        read_val_clean, _  = read_one(bank, vars_clean, h_target)
        read_val_dist, _  = read_one(bank, vars, h_target)
        expected_v, _= bank.apply(vars, h_target, method=lambda mdl, x: mdl.v_proj(x), mutable=['memory'])
        
        sim_clean = cosine_float(read_val_clean[0], expected_v[0])
        sim_dist = cosine_float(read_val_dist[0], expected_v[0])
        sims.append((sim_clean, sim_dist))

    sim_degrades = [c - d for c, d in sims]
    mean_degrade = np.mean(sim_degrades)
    # Expect degradation to be relatively bounded (not complete catastrophic forgetting)
    passed = mean_degrade < 0.5
    
    result = {"sims": sims, "mean_degrade": mean_degrade, "pass": passed}
    print(f"  Mean similarity degradation = {mean_degrade:.4f}  → {'PASS' if passed else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 5: Capacity Scaling
# ---------------------------------------------------------------------------
def test5_capacity_scaling():
    print("\n[TEST 5] Capacity Scaling")
    capacities = [16, 32, 64, 128]
    results = {}

    for cap in capacities:
        config = make_config(capacity=cap, top_k=4, write_threshold=0.9, dim=16, hidden=16)
        bank, vars = fresh_bank(config, seed=5)
        rng = jax.random.PRNGKey(500)

        n_writes = min(cap, 20)
        for i in range(n_writes):
            rng, k = jax.random.split(rng)
            h = jax.random.normal(k, (1, config.hidden_size))
            vars = write_one(bank, vars, h)

        active = int(jnp.sum(vars['memory']['state'] == STATE_ACTIVE))
        results[cap] = {"active": active, "written": n_writes,
                        "pass": active >= int(n_writes * 0.8)}
        print(f"  capacity={cap:3d}: written={n_writes} active={active}  → {'PASS' if results[cap]['pass'] else 'FAIL'}")

    return results


# ---------------------------------------------------------------------------
# Test 6: Replacement Policy
# ---------------------------------------------------------------------------
def test6_replacement():
    print("\n[TEST 6] Replacement Policy")
    cap = 4
    config = make_config(capacity=cap, write_threshold=1.1) # force insert
    bank, vars = fresh_bank(config, seed=6)
    rng = jax.random.PRNGKey(600)

    # Fill to capacity
    for i in range(cap):
        rng, k = jax.random.split(rng)
        h = jax.random.normal(k, (1, config.hidden_size))
        vars = write_one(bank, vars, h, wp=jnp.array([2.0]))

    unfrozen = flax.core.unfreeze(vars)
    unfrozen['memory']['state'] = jnp.full((cap,), STATE_ACTIVE, dtype=jnp.int32)
    # Slot 0 has lowest importance
    unfrozen['memory']['importance'] = jnp.array([0.1, 0.5, 0.9, 0.7], dtype=jnp.float32)
    vars = flax.core.freeze(unfrozen)

    keys_before = np.array(vars['memory']['keys'])

    # Write one more
    rng, k = jax.random.split(rng)
    h_new = jax.random.normal(k, (1, config.hidden_size))
    vars  = write_one(bank, vars, h_new, wp=jnp.array([2.0]))

    keys_after = np.array(vars['memory']['keys'])
    
    # Assert exact replacement
    diff = np.sum((keys_before - keys_after)**2, axis=1)
    slot0_changed = diff[0] > 1e-5
    other_unchanged = np.all(diff[1:] < 1e-5)

    passed = slot0_changed and other_unchanged
    result = {"slot0_changed": slot0_changed, "other_unchanged": other_unchanged, "pass": passed}
    print(f"  Exact slot replaced (lowest importance): {passed}  → {'PASS' if passed else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 7: Recency Effect
# ---------------------------------------------------------------------------
def test7_recency():
    print("\n[TEST 7] Recency Effect")
    config = make_config(top_k=1)
    bank, vars = fresh_bank(config, seed=7)
    
    # Inject two identical slots except for recency
    unfrozen = flax.core.unfreeze(vars)
    unfrozen['memory']['keys'] = unfrozen['memory']['keys'].at[0:2].set(jnp.ones((2, config.memory_dim)))
    unfrozen['memory']['vals'] = unfrozen['memory']['vals'].at[0].set(jnp.full((config.memory_dim,), 1.0))
    unfrozen['memory']['vals'] = unfrozen['memory']['vals'].at[1].set(jnp.full((config.memory_dim,), 2.0))
    unfrozen['memory']['state'] = unfrozen['memory']['state'].at[0:2].set(STATE_ACTIVE)
    unfrozen['memory']['importance'] = jnp.zeros_like(unfrozen['memory']['importance'])
    unfrozen['memory']['confidence'] = jnp.zeros_like(unfrozen['memory']['confidence'])
    
    unfrozen['memory']['last_access'] = unfrozen['memory']['last_access'].at[0].set(10) # older
    unfrozen['memory']['last_access'] = unfrozen['memory']['last_access'].at[1].set(100) # newer
    unfrozen['memory']['global_step'] = jnp.array(100, dtype=jnp.int32)
    vars = flax.core.freeze(unfrozen)

    # Query with exact key
    # We must construct a query_h that perfectly maps to key=1.0 via q_proj. 
    # Actually, we can bypass q_proj if we use the same key for both slots and an ambiguous query.
    # Since both slots have the exact same key (all ones), they will have identical cosine sim.
    # The only difference is recency. Slot 1 is newer, so it should be retrieved.
    h_query = jax.random.normal(jax.random.PRNGKey(77), (1, config.hidden_size))
    read_val, _ = read_one(bank, vars, h_query)
    
    # Since top_k=1, read_val will be a weighted sum of JUST slot 1 (val=2.0)
    # We check if the mean value is closer to 2.0 than 1.0.
    val_mean = float(jnp.mean(read_val))
    passed = abs(val_mean - 2.0) < abs(val_mean - 1.0)

    result = {"val_mean": val_mean, "pass": passed}
    print(f"  Recency-driven retrieval value={val_mean:.4f} (expected ~2.0) → {'PASS' if passed else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 8: Importance Effect
# ---------------------------------------------------------------------------
def test8_importance():
    print("\n[TEST 8] Importance Effect")
    config = make_config(top_k=1)
    bank, vars = fresh_bank(config, seed=8)

    unfrozen = flax.core.unfreeze(vars)
    unfrozen['memory']['keys'] = unfrozen['memory']['keys'].at[0:2].set(jnp.ones((2, config.memory_dim)))
    unfrozen['memory']['vals'] = unfrozen['memory']['vals'].at[0].set(jnp.full((config.memory_dim,), 1.0))
    unfrozen['memory']['vals'] = unfrozen['memory']['vals'].at[1].set(jnp.full((config.memory_dim,), 2.0))
    unfrozen['memory']['state'] = unfrozen['memory']['state'].at[0:2].set(STATE_ACTIVE)
    unfrozen['memory']['last_access'] = jnp.zeros_like(unfrozen['memory']['last_access'])
    unfrozen['memory']['confidence'] = jnp.zeros_like(unfrozen['memory']['confidence'])
    
    # Slot 0 has higher importance
    unfrozen['memory']['importance'] = unfrozen['memory']['importance'].at[0].set(0.9)
    unfrozen['memory']['importance'] = unfrozen['memory']['importance'].at[1].set(0.1)
    vars = flax.core.freeze(unfrozen)

    h_query = jax.random.normal(jax.random.PRNGKey(88), (1, config.hidden_size))
    read_val, _ = read_one(bank, vars, h_query)
    
    val_mean = float(jnp.mean(read_val))
    passed = abs(val_mean - 1.0) < abs(val_mean - 2.0)

    result = {"val_mean": val_mean, "pass": passed}
    print(f"  Importance-driven retrieval value={val_mean:.4f} (expected ~1.0) → {'PASS' if passed else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 9: Confidence Effect
# ---------------------------------------------------------------------------
def test9_confidence():
    print("\n[TEST 9] Confidence Effect")
    config = make_config(top_k=1)
    bank, vars = fresh_bank(config, seed=9)

    unfrozen = flax.core.unfreeze(vars)
    unfrozen['memory']['keys'] = unfrozen['memory']['keys'].at[0:2].set(jnp.ones((2, config.memory_dim)))
    unfrozen['memory']['vals'] = unfrozen['memory']['vals'].at[0].set(jnp.full((config.memory_dim,), 1.0))
    unfrozen['memory']['vals'] = unfrozen['memory']['vals'].at[1].set(jnp.full((config.memory_dim,), 2.0))
    unfrozen['memory']['state'] = unfrozen['memory']['state'].at[0:2].set(STATE_ACTIVE)
    unfrozen['memory']['last_access'] = jnp.zeros_like(unfrozen['memory']['last_access'])
    unfrozen['memory']['importance'] = jnp.zeros_like(unfrozen['memory']['importance'])
    
    # Slot 1 has higher confidence
    unfrozen['memory']['confidence'] = unfrozen['memory']['confidence'].at[0].set(0.1)
    unfrozen['memory']['confidence'] = unfrozen['memory']['confidence'].at[1].set(0.9)
    vars = flax.core.freeze(unfrozen)

    h_query = jax.random.normal(jax.random.PRNGKey(99), (1, config.hidden_size))
    read_val, _ = read_one(bank, vars, h_query)
    
    val_mean = float(jnp.mean(read_val))
    passed = abs(val_mean - 2.0) < abs(val_mean - 1.0)

    result = {"val_mean": val_mean, "pass": passed}
    print(f"  Confidence-driven retrieval value={val_mean:.4f} (expected ~2.0) → {'PASS' if passed else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 10: Forgetting (ACTIVE → DORMANT → EXPIRED)
# ---------------------------------------------------------------------------
def test10_forgetting():
    print("\n[TEST 10] Forgetting (Decay State Transitions)")
    config = make_config(decay_rate=1.0)  # fast decay for test
    bank, vars = fresh_bank(config, seed=10)

    h = jax.random.normal(jax.random.PRNGKey(1000), (1, config.hidden_size))
    vars = write_one(bank, vars, h)

    initial_state = int(vars['memory']['state'][0])

    # dt=0 → still ACTIVE (no time passed)
    _, v0 = bank.apply(vars, method=bank.decay_memory, mutable=['memory'])
    s0    = int(v0['memory']['state'][0])

    # dt=1 → R=exp(-1)≈0.368 < 0.5 → DORMANT
    unfrozen = flax.core.unfreeze(vars)
    unfrozen['memory']['global_step'] = jnp.array(1, dtype=jnp.int32)
    vars_t1 = flax.core.freeze(unfrozen)
    _, v1 = bank.apply(vars_t1, method=bank.decay_memory, mutable=['memory'])
    s1    = int(v1['memory']['state'][0])

    # dt=10000 → R≈0 → EXPIRED
    unfrozen = flax.core.unfreeze(vars)
    unfrozen['memory']['global_step'] = jnp.array(10000, dtype=jnp.int32)
    vars_t2 = flax.core.freeze(unfrozen)
    _, v2 = bank.apply(vars_t2, method=bank.decay_memory, mutable=['memory'])
    s2    = int(v2['memory']['state'][0])

    state_names = {0: "EXPIRED", 1: "ACTIVE", 2: "DORMANT"}
    transitions  = f"initial={state_names[initial_state]} → dt=1:{state_names[s1]} → dt=10000:{state_names[s2]}"
    passed = (s1 in (STATE_DORMANT, STATE_EXPIRED)) and (s2 == STATE_EXPIRED)

    result = {"initial": state_names[initial_state],
              "after_dt1": state_names[s1],
              "after_dt10000": state_names[s2],
              "transitions": transitions,
              "pass": passed}
    print(f"  {transitions}  → {'PASS' if passed else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 11: Counterfactual Causal Test
# ---------------------------------------------------------------------------
def test11_counterfactual():
    print("\n[TEST 11] Counterfactual Causal Test")
    config = make_config()
    bank, vars = fresh_bank(config, seed=11)

    K = jax.random.normal(jax.random.PRNGKey(1100), (1, config.hidden_size))
    K = K / jnp.linalg.norm(K)

    # Exp A: write K with original params
    vars_A = {'params': vars['params'], 'memory': {
        **{k: jnp.zeros_like(v) if v.dtype != jnp.int32 else jnp.zeros_like(v)
           for k, v in vars['memory'].items()},
        'state': jnp.full((config.memory_capacity,), STATE_EXPIRED, jnp.int32),
        'global_step': jnp.zeros((), jnp.int32),
    }}
    vars_A = write_one(bank, vars_A, K)
    R_A, _ = read_one(bank, vars_A, K)

    # Exp B: negate v_proj → V_B = -V_A for same K
    unfrozen = flax.core.unfreeze(vars)
    unfrozen['params']['v_proj']['kernel'] = -unfrozen['params']['v_proj']['kernel']
    vars_B_params = flax.core.freeze(unfrozen)

    vars_B = {'params': vars_B_params['params'], 'memory': {
        **{k: jnp.zeros_like(v) for k, v in vars['memory'].items()
           if v.dtype != jnp.int32},
        **{k: jnp.zeros_like(v) for k, v in vars['memory'].items()
           if v.dtype == jnp.int32},
        'state': jnp.full((config.memory_capacity,), STATE_EXPIRED, jnp.int32),
        'global_step': jnp.zeros((), jnp.int32),
    }}
    # Simpler: fresh memory dict
    mem_empty = {
        'keys':         jnp.zeros((config.memory_capacity, config.memory_dim)),
        'vals':         jnp.zeros((config.memory_capacity, config.memory_dim)),
        'importance':   jnp.zeros((config.memory_capacity,)),
        'confidence':   jnp.zeros((config.memory_capacity,)),
        'created_at':   jnp.zeros((config.memory_capacity,), jnp.int32),
        'last_access':  jnp.zeros((config.memory_capacity,), jnp.int32),
        'access_count': jnp.zeros((config.memory_capacity,), jnp.int32),
        'state':        jnp.full((config.memory_capacity,), STATE_EXPIRED, jnp.int32),
        'global_step':  jnp.zeros((), jnp.int32),
    }
    vars_B = {'params': vars_B_params['params'], 'memory': mem_empty}
    vars_B = write_one(bank, vars_B, K)
    R_B, _ = read_one(bank, vars_B, K)

    sim_cross = cosine_float(np.array(R_A[0]), np.array(R_B[0]))
    passed    = sim_cross < 0.99  # different values → different retrieval

    result = {"cross_sim": sim_cross,
              "causal_effect": sim_cross < 0.99,
              "pass": passed}
    print(f"  Cross-similarity (RA vs RB) = {sim_cross:.4f}  → {'PASS (causal effect detected)' if passed else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_all_tests():
    print("=" * 60)
    print("  MEMORY BANK FUNCTIONAL BENCHMARK (11 Tests)")
    print("=" * 60)

    results = {}
    results['test1']  = test1_basic_write()
    results['test2']  = test2_basic_read()
    results['test3']  = test3_distractor_retrieval()
    results['test4']  = test4_interference()
    results['test5']  = test5_capacity_scaling()
    results['test6']  = test6_replacement()
    results['test7']  = test7_recency()
    results['test8']  = test8_importance()
    results['test9']  = test9_confidence()
    results['test10'] = test10_forgetting()
    results['test11'] = test11_counterfactual()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    top_level_pass = {
        'test1':  results['test1']['pass'],
        'test2':  results['test2']['pass'],
        'test3':  results['test3']['pass'],
        'test4':  results['test4']['pass'],
        'test5':  all(v['pass'] for v in results['test5'].values()),
        'test6':  results['test6']['pass'],
        'test7':  results['test7']['pass'],
        'test8':  results['test8']['pass'],
        'test9':  results['test9']['pass'],
        'test10': results['test10']['pass'],
        'test11': results['test11']['pass'],
    }

    names = {
        'test1': 'Basic Write',
        'test2': 'Basic Read',
        'test3': 'Distractor Retrieval',
        'test4': 'Interference',
        'test5': 'Capacity Scaling',
        'test6': 'Replacement Policy',
        'test7': 'Recency Effect',
        'test8': 'Importance Effect',
        'test9': 'Confidence Effect',
        'test10':'Forgetting',
        'test11':'Counterfactual',
    }

    all_pass = True
    for k, v in top_level_pass.items():
        status = "PASS" if v else "FAIL"
        print(f"  {names[k]:30s} {status}")
        if not v: all_pass = False

    print("-" * 60)
    print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return results


if __name__ == '__main__':
    run_all_tests()
