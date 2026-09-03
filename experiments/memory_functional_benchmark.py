"""
experiments/memory_functional_benchmark.py (PyTorch Version)

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

import torch
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
    torch.manual_seed(seed)
    bank = TinyMemoryBank(config=config)
    bank.load_memory_state(bank.empty_memory_state())
    return bank


def write_one(bank, h, eos=None, wp=None):
    if eos is None:
        eos = torch.ones(h.shape[0], device=h.device)
    if wp is None:
        wp = torch.ones(h.shape[0], device=h.device)
    bank.write(h, eos, wp)
    return bank


def read_one(bank, h):
    out = bank.read(h)
    return out


def cosine_float(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a / na, b / nb))


# ---------------------------------------------------------------------------
# Test 1: Basic Write
# ---------------------------------------------------------------------------
def test1_basic_write():
    print("\n[TEST 1] Basic Write")
    config = make_config()
    bank = fresh_bank(config, seed=1)

    torch.manual_seed(1)
    h = torch.randn(1, config.hidden_size)
    write_one(bank, h)

    active = int(torch.sum(bank.mem_state == STATE_ACTIVE))
    result = {"active_after_write": active, "expected": 1, "pass": active == 1}
    print(f"  active_count after 1 write = {active}  → {'PASS' if result['pass'] else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 2: Basic Read
# ---------------------------------------------------------------------------
def test2_basic_read():
    print("\n[TEST 2] Basic Read")
    config = make_config()
    bank = fresh_bank(config, seed=2)

    torch.manual_seed(2)
    h = torch.randn(1, config.hidden_size)
    h = h / torch.norm(h)

    write_one(bank, h)
    R = read_one(bank, h)
    expected_v = bank.v_proj(h)

    sim = cosine_float(R[0].detach().numpy(), expected_v[0].detach().numpy())
    passed = sim > 0.1
    result = {"cosine_sim": sim, "pass": passed}
    print(f"  Write-Read cosine similarity = {sim:.4f}  → {'PASS' if passed else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 3: Distractor Retrieval
# ---------------------------------------------------------------------------
def test3_distractor_retrieval(n_distractors=10, seeds=3):
    print(f"\n[TEST 3] Distractor Retrieval (N={n_distractors})")
    config = make_config(capacity=64, top_k=8, write_threshold=0.9)

    r1_scores, mrr_scores = [], []
    for seed in range(seeds):
        bank = fresh_bank(config, seed=seed)
        with torch.no_grad():
            bank.q_proj.weight.data.copy_(bank.k_proj.weight.data)
        torch.manual_seed(seed + 100)

        h_target = torch.randn(1, config.hidden_size)
        h_target = h_target / torch.norm(h_target)
        write_one(bank, h_target)
        gt_idx = 0

        for _ in range(n_distractors):
            h_d = torch.randn(1, config.hidden_size)
            h_d = h_d / torch.norm(h_d)
            write_one(bank, h_d)

        # Retrieve scores using actual bank scoring
        _, actual_scores = bank.read(h_target, return_scores=True)
        scores = actual_scores[0].detach().numpy()

        r1 = recall_at_k(scores, gt_idx, k_values=[1, config.memory_top_k])
        mrr = mean_reciprocal_rank(scores, gt_idx)
        r1_scores.append(r1[1])
        mrr_scores.append(mrr)

    result = {
        "Recall@1": np.mean(r1_scores),
        "MRR": np.mean(mrr_scores),
        "pass": np.mean(r1_scores) >= 0.5,
    }
    print(f"  Recall@1={result['Recall@1']:.4f}  MRR={result['MRR']:.4f}  → {'PASS' if result['pass'] else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 4: Interference
# ---------------------------------------------------------------------------
def test4_interference(n_distractors=20, seeds=3):
    print(f"\n[TEST 4] Interference (N={n_distractors})")
    config = make_config(capacity=64, top_k=8, write_threshold=0.9)

    clean_r1, dist_r1 = [], []
    for seed in range(seeds):
        bank = fresh_bank(config, seed=seed)
        with torch.no_grad():
            bank.q_proj.weight.data.copy_(bank.k_proj.weight.data)
        torch.manual_seed(seed + 200)

        h_target = torch.randn(1, config.hidden_size)
        h_target = h_target / torch.norm(h_target)
        write_one(bank, h_target)
        gt_idx = 0

        def get_r1(b):
            _, scores_t = b.read(h_target, return_scores=True)
            scores = scores_t[0].detach().numpy()
            r1 = recall_at_k(scores, gt_idx, k_values=[1])
            return r1[1]

        clean_r1.append(get_r1(bank))

        for _ in range(n_distractors):
            h_d = torch.randn(1, config.hidden_size)
            h_d = h_d / torch.norm(h_d)
            write_one(bank, h_d)

        dist_r1.append(get_r1(bank))

    mean_clean_r1 = np.mean(clean_r1)
    mean_dist_r1 = np.mean(dist_r1)
    random_baseline = 1.0 / (n_distractors + 1)
    
    passed = mean_dist_r1 > random_baseline * 2
    result = {"clean_r1": mean_clean_r1, "dist_r1": mean_dist_r1, "random": random_baseline, "pass": passed}
    print(f"  Recall@1 Clean={mean_clean_r1:.2f}, Distractor={mean_dist_r1:.2f} (Random={random_baseline:.4f})  → {'PASS' if passed else 'FAIL'}")
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
        bank = fresh_bank(config, seed=5)
        torch.manual_seed(500)

        n_writes = min(cap, 20)
        for _ in range(n_writes):
            h = torch.randn(1, config.hidden_size)
            write_one(bank, h)

        active = int(torch.sum(bank.mem_state == STATE_ACTIVE))
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
    config = make_config(capacity=cap, write_threshold=1.1)  # force insert
    bank = fresh_bank(config, seed=6)
    torch.manual_seed(600)

    # Fill to capacity
    for _ in range(cap):
        h = torch.randn(1, config.hidden_size)
        write_one(bank, h, wp=torch.tensor([2.0]))

    bank.mem_state.fill_(STATE_ACTIVE)
    # Slot 0 has lowest importance
    bank.mem_importance.copy_(torch.tensor([0.1, 0.5, 0.9, 0.7]))
    bank.mem_confidence.fill_(0.9)

    keys_before = bank.mem_keys.clone()
    vals_before = bank.mem_vals.clone()
    imp_before = bank.mem_importance.clone()
    conf_before = bank.mem_confidence.clone()
    created_before = bank.mem_created_at.clone()
    last_acc_before = bank.mem_last_access.clone()
    acc_cnt_before = bank.mem_access_count.clone()

    # Step forward in time before writing one more
    bank.global_step[0] = 10

    # Write one more
    h_new = torch.randn(1, config.hidden_size)
    write_one(bank, h_new, wp=torch.tensor([2.0]))

    keys_after = bank.mem_keys.clone()
    vals_after = bank.mem_vals.clone()
    imp_after = bank.mem_importance.clone()
    conf_after = bank.mem_confidence.clone()
    created_after = bank.mem_created_at.clone()
    last_acc_after = bank.mem_last_access.clone()
    acc_cnt_after = bank.mem_access_count.clone()

    diff_keys = torch.sum((keys_before - keys_after) ** 2, dim=1)
    diff_vals = torch.sum((vals_before - vals_after) ** 2, dim=1)

    slot0_changed = (diff_keys[0] > 1e-5) and (diff_vals[0] > 1e-5) and \
                    (imp_before[0] != imp_after[0]) and (conf_before[0] != conf_after[0]) and \
                    (created_before[0] != created_after[0]) and (last_acc_before[0] != last_acc_after[0])

    other_unchanged = torch.all(diff_keys[1:] < 1e-5) and torch.all(diff_vals[1:] < 1e-5) and \
                      torch.all(imp_before[1:] == imp_after[1:]) and torch.all(conf_before[1:] == conf_after[1:]) and \
                      torch.all(created_before[1:] == created_after[1:]) and torch.all(last_acc_before[1:] == last_acc_after[1:]) and \
                      torch.all(acc_cnt_before[1:] == acc_cnt_after[1:])

    passed = bool(slot0_changed and other_unchanged)
    result = {"slot0_changed": bool(slot0_changed), "other_unchanged": bool(other_unchanged), "pass": passed}
    print(f"  Exact slot replaced (ALL METADATA for lowest importance): {passed}  → {'PASS' if passed else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 7: Recency Effect
# ---------------------------------------------------------------------------
def test7_recency():
    print("\n[TEST 7] Recency Effect")
    config = make_config(top_k=1)
    bank = fresh_bank(config, seed=7)

    bank.mem_keys[0:2] = 1.0
    bank.mem_vals[0] = 1.0
    bank.mem_vals[1] = 2.0
    bank.mem_state[0:2] = STATE_ACTIVE
    bank.mem_importance.zero_()
    bank.mem_confidence.zero_()

    bank.mem_last_access[0] = 10   # older
    bank.mem_last_access[1] = 100  # newer
    bank.global_step[0] = 100

    torch.manual_seed(77)
    h_query = torch.randn(1, config.hidden_size)
    read_val = read_one(bank, h_query)

    val_mean = float(torch.mean(read_val))
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
    bank = fresh_bank(config, seed=8)

    bank.mem_keys[0:2] = 1.0
    bank.mem_vals[0] = 1.0
    bank.mem_vals[1] = 2.0
    bank.mem_state[0:2] = STATE_ACTIVE
    bank.mem_last_access.zero_()
    bank.mem_confidence.zero_()

    bank.mem_importance[0] = 0.9
    bank.mem_importance[1] = 0.1

    torch.manual_seed(88)
    h_query = torch.randn(1, config.hidden_size)
    read_val = read_one(bank, h_query)

    val_mean = float(torch.mean(read_val))
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
    bank = fresh_bank(config, seed=9)

    bank.mem_keys[0:2] = 1.0
    bank.mem_vals[0] = 1.0
    bank.mem_vals[1] = 2.0
    bank.mem_state[0:2] = STATE_ACTIVE
    bank.mem_last_access.zero_()
    bank.mem_importance.zero_()

    bank.mem_confidence[0] = 0.1
    bank.mem_confidence[1] = 0.9

    torch.manual_seed(99)
    h_query = torch.randn(1, config.hidden_size)
    read_val = read_one(bank, h_query)

    val_mean = float(torch.mean(read_val))
    passed = abs(val_mean - 2.0) < abs(val_mean - 1.0)

    result = {"val_mean": val_mean, "pass": passed}
    print(f"  Confidence-driven retrieval value={val_mean:.4f} (expected ~2.0) → {'PASS' if passed else 'FAIL'}")
    return result


# ---------------------------------------------------------------------------
# Test 10: Forgetting (ACTIVE → DORMANT → EXPIRED)
# ---------------------------------------------------------------------------
def test10_forgetting():
    print("\n[TEST 10] Forgetting (Decay State Transitions)")
    config = make_config(decay_rate=1.0)
    bank = fresh_bank(config, seed=10)

    torch.manual_seed(1000)
    h = torch.randn(1, config.hidden_size)
    write_one(bank, h)

    initial_state = int(bank.mem_state[0])

    # dt=0 → still ACTIVE
    bank.decay_memory()
    s0 = int(bank.mem_state[0])

    # dt=1 → DORMANT
    bank.global_step[0] = 1
    bank.decay_memory()
    s1 = int(bank.mem_state[0])

    # dt=10000 → EXPIRED
    bank.global_step[0] = 10000
    bank.decay_memory()
    s2 = int(bank.mem_state[0])

    state_names = {0: "EXPIRED", 1: "ACTIVE", 2: "DORMANT"}
    transitions = f"initial={state_names[initial_state]} → dt=1:{state_names[s1]} → dt=10000:{state_names[s2]}"
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
    bank = fresh_bank(config, seed=11)

    torch.manual_seed(1100)
    K = torch.randn(1, config.hidden_size)
    K = K / torch.norm(K)

    # Exp A
    bank.load_memory_state(bank.empty_memory_state())
    write_one(bank, K)
    R_A = read_one(bank, K).clone()

    # Exp B: negate v_proj
    with torch.no_grad():
        bank.v_proj.weight.data.neg_()

    bank.load_memory_state(bank.empty_memory_state())
    write_one(bank, K)
    R_B = read_one(bank, K).clone()

    # Restore v_proj
    with torch.no_grad():
        bank.v_proj.weight.data.neg_()

    sim_cross = cosine_float(R_A[0].detach().numpy(), R_B[0].detach().numpy())
    passed = sim_cross < 0.99

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
    print("  MEMORY BANK FUNCTIONAL BENCHMARK (11 Tests - PyTorch)")
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
        if not v:
            all_pass = False

    print("-" * 60)
    print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return results


if __name__ == '__main__':
    run_all_tests()
