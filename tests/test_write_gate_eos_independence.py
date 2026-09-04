"""
tests/test_write_gate_eos_independence.py

Mandatory tests for the is_eos removal change:
- EOS-independence: toggling is_eos alone does NOT change write decision
- Causal validation: write_prob crossing threshold DOES change write decision
- Write-collapse detection
- Duplicate-write / reinforcement behavior
- Anti-regression: replacement still uses key/value/metadata identity
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.tiny_memory_bank import (
    TinyMemoryBank, TinyMemoryConfig,
    STATE_ACTIVE, STATE_EXPIRED
)


@pytest.fixture
def gate_bank():
    config = TinyMemoryConfig(
        memory_capacity=16, memory_dim=16, hidden_size=16,
        memory_top_k=4, mem_decay_rate=0.0,
        memory_write_threshold=0.5, memory_read_threshold=0.0,
    )
    torch.manual_seed(42)
    bank = TinyMemoryBank(config=config)
    bank.load_memory_state(bank.empty_memory_state())
    return bank, config


def snap(bank):
    return {k: getattr(bank, f'mem_{k}' if k != 'global_step' else k).clone()
            for k in ['keys', 'vals', 'importance', 'confidence',
                      'created_at', 'last_access', 'access_count', 'state']}


def states_equal(s1, s2):
    return all(torch.equal(s1[k], s2[k]) for k in s1)


class TestEOSIndependence:
    def test_eos_true_vs_false_both_write_when_prob_above_threshold(self, gate_bank):
        """is_eos=True and is_eos=False must both WRITE with high write_prob."""
        bank, config = gate_bank
        torch.manual_seed(100)
        h = torch.randn(1, config.hidden_size)
        wp = torch.full((1,), 0.9)

        bank.load_memory_state(bank.empty_memory_state())
        idx_true = bank.write(h.clone(), torch.ones(1), wp.clone())
        state_eos_true = snap(bank)

        bank.load_memory_state(bank.empty_memory_state())
        idx_false = bank.write(h.clone(), torch.zeros(1), wp.clone())
        state_eos_false = snap(bank)

        assert idx_true[0].item() != -1, "is_eos=True + high prob must write"
        assert idx_false[0].item() != -1, "is_eos=False + high prob must write"
        assert idx_true[0].item() == idx_false[0].item(), (
            f"Same h+wp must target same slot. Got {idx_true[0].item()} vs {idx_false[0].item()}")
        assert states_equal(state_eos_true, state_eos_false), \
            "Memory state must be identical regardless of is_eos"

    def test_eos_true_vs_false_both_no_write_when_prob_below_threshold(self, gate_bank):
        """is_eos=True and is_eos=False must both NOT WRITE with low write_prob."""
        bank, config = gate_bank
        torch.manual_seed(101)
        h = torch.randn(1, config.hidden_size)
        wp = torch.full((1,), 0.1)
        bank.load_memory_state(bank.empty_memory_state())
        before = snap(bank)
        idx_true = bank.write(h.clone(), torch.ones(1), wp.clone())
        bank.load_memory_state(bank.empty_memory_state())
        idx_false = bank.write(h.clone(), torch.zeros(1), wp.clone())
        assert idx_true[0].item() == -1 and idx_false[0].item() == -1

    def test_eos_independence_stored_key_value_metadata(self, gate_bank):
        """Stored key, value, importance, confidence must be identical for eos=True vs False."""
        bank, config = gate_bank
        torch.manual_seed(102)
        h = torch.randn(1, config.hidden_size)
        wp = torch.ones(1)

        bank.load_memory_state(bank.empty_memory_state())
        idx_t = bank.write(h.clone(), torch.ones(1), wp.clone())
        s = idx_t[0].item()
        key_t, val_t = bank.mem_keys[s].clone(), bank.mem_vals[s].clone()
        imp_t, conf_t = bank.mem_importance[s].clone(), bank.mem_confidence[s].clone()

        bank.load_memory_state(bank.empty_memory_state())
        idx_f = bank.write(h.clone(), torch.zeros(1), wp.clone())
        sf = idx_f[0].item()
        key_f, val_f = bank.mem_keys[sf].clone(), bank.mem_vals[sf].clone()
        imp_f, conf_f = bank.mem_importance[sf].clone(), bank.mem_confidence[sf].clone()

        assert s == sf
        assert torch.allclose(key_t, key_f, atol=1e-6)
        assert torch.allclose(val_t, val_f, atol=1e-6)
        assert torch.allclose(imp_t, imp_f, atol=1e-6)
        assert torch.allclose(conf_t, conf_f, atol=1e-6)


class TestWriteProbCausal:
    def test_write_prob_below_threshold_no_write(self, gate_bank):
        bank, config = gate_bank
        torch.manual_seed(200)
        h = torch.randn(1, config.hidden_size)
        wp = torch.full((1,), config.memory_write_threshold - 0.001)
        before = snap(bank)
        idx = bank.write(h, torch.zeros(1), wp)
        assert idx[0].item() == -1
        assert states_equal(snap(bank), before)

    def test_write_prob_at_threshold_writes(self, gate_bank):
        bank, config = gate_bank
        torch.manual_seed(201)
        h = torch.randn(1, config.hidden_size)
        idx = bank.write(h, torch.zeros(1), torch.full((1,), config.memory_write_threshold))
        assert idx[0].item() != -1

    def test_write_prob_above_threshold_writes(self, gate_bank):
        bank, config = gate_bank
        torch.manual_seed(202)
        h = torch.randn(1, config.hidden_size)
        idx = bank.write(h, torch.zeros(1), torch.full((1,), config.memory_write_threshold + 0.001))
        assert idx[0].item() != -1

    def test_is_eos_change_alone_does_not_flip_decision(self, gate_bank):
        """With write_prob constant above threshold, flipping is_eos must not change outcome."""
        bank, config = gate_bank
        torch.manual_seed(203)
        h = torch.randn(1, config.hidden_size)
        wp = torch.full((1,), 0.9)
        bank.load_memory_state(bank.empty_memory_state())
        idx_0 = bank.write(h.clone(), torch.zeros(1), wp.clone())
        bank.load_memory_state(bank.empty_memory_state())
        idx_1 = bank.write(h.clone(), torch.ones(1), wp.clone())
        both_wrote = idx_0[0].item() != -1 and idx_1[0].item() != -1
        neither = idx_0[0].item() == -1 and idx_1[0].item() == -1
        assert both_wrote or neither, (
            f"Flipping is_eos alone must not change decision: "
            f"eos=0 → {idx_0[0].item()}, eos=1 → {idx_1[0].item()}")


class TestWriteCollapse:
    def test_no_collapse_with_write_prob_one_eos_false(self):
        config = TinyMemoryConfig(memory_capacity=16, memory_dim=16, hidden_size=16,
            memory_top_k=4, mem_decay_rate=0.0, memory_write_threshold=0.0)
        torch.manual_seed(300)
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())
        n = 16
        ok = sum(1 for _ in range(n)
                 if bank.write(torch.randn(1,16), torch.zeros(1), torch.ones(1))[0].item() != -1)
        write_rate = ok / n
        print(f"\n  write_rate={write_rate:.2%} ({ok}/{n})")
        assert write_rate > 0.5, f"Write collapse detected: rate={write_rate:.2%}"

    def test_no_uncontrolled_write_below_threshold(self):
        config = TinyMemoryConfig(memory_capacity=16, memory_dim=16, hidden_size=16,
            memory_top_k=4, mem_decay_rate=0.0, memory_write_threshold=0.9)
        torch.manual_seed(301)
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())
        ok = sum(1 for _ in range(16)
                 if bank.write(torch.randn(1,16), torch.zeros(1), torch.zeros(1))[0].item() != -1)
        assert ok == 0, f"Uncontrolled write: {ok} wrote with wp<threshold"

    def test_write_rate_report(self):
        config = TinyMemoryConfig(memory_capacity=32, memory_dim=16, hidden_size=16,
            memory_top_k=4, mem_decay_rate=0.001, memory_write_threshold=0.0)
        torch.manual_seed(302)
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())
        n, ok = 64, 0
        for i in range(n):
            wp = torch.ones(1) if i % 5 != 0 else torch.zeros(1)
            if bank.write(torch.randn(1,16), torch.zeros(1), wp)[0].item() != -1:
                ok += 1
        wr = ok / n
        occ = int(torch.sum(bank.mem_state != 0).item()) / config.memory_capacity
        print(f"\n  total={n}, successful={ok}, write_rate={wr:.2%}, occupancy={occ:.2%}")
        assert wr > 0.0 and wr <= 1.0


class TestDuplicateWrite:
    def test_repeated_same_fact_updates_not_inserts(self):
        config = TinyMemoryConfig(memory_capacity=16, memory_dim=16, hidden_size=16,
            memory_top_k=4, mem_decay_rate=0.0, memory_write_threshold=0.0)
        torch.manual_seed(400)
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())
        h = torch.randn(1, config.hidden_size)
        bank.write(h.clone(), torch.zeros(1), torch.ones(1))
        n1 = int(torch.sum(bank.mem_state == STATE_ACTIVE))
        bank.write(h.clone(), torch.zeros(1), torch.ones(1))
        n2 = int(torch.sum(bank.mem_state == STATE_ACTIVE))
        assert n1 == 1 and n2 == 1, "Duplicate must UPDATE, not insert new slot"

    def test_repeated_write_increases_confidence(self):
        config = TinyMemoryConfig(memory_capacity=16, memory_dim=16, hidden_size=16,
            memory_top_k=4, mem_decay_rate=0.0, memory_write_threshold=0.0)
        torch.manual_seed(401)
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())
        h = torch.randn(1, config.hidden_size)
        idx1 = bank.write(h.clone(), torch.zeros(1), torch.ones(1))
        slot = idx1[0].item()
        conf1 = float(bank.mem_confidence[slot])
        bank.write(h.clone(), torch.zeros(1), torch.ones(1))
        conf2 = float(bank.mem_confidence[slot])
        assert conf2 > conf1, f"Confidence must increase: {conf1:.3f} -> {conf2:.3f}"


class TestReplacementIntegrity:
    def test_replacement_changes_only_target_slot(self):
        """
        After capacity fill, verify replacement changes ONLY the target slot key.
        Uses write_threshold=0.9 so random vectors always INSERT (not UPDATE).
        """
        config = TinyMemoryConfig(memory_capacity=4, memory_dim=8, hidden_size=8,
            memory_top_k=2, mem_decay_rate=0.0,
            memory_write_threshold=0.9)  # high threshold → always INSERT for random vectors
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())

        # Use orthogonal-ish vectors to guarantee 4 distinct INSERTs
        g = torch.Generator()
        g.manual_seed(500)
        for i in range(4):
            # Use basis-like vectors with noise to stay below cosine threshold
            h = torch.zeros(1, 8)
            h[0, i % 8] = 1.0
            h = h + torch.randn(1, 8, generator=g) * 0.01
            bank.write(h, torch.zeros(1), torch.ones(1))

        n_active = int(torch.sum(bank.mem_state == STATE_ACTIVE))
        assert n_active == 4, f"Expected 4 active slots after 4 INSERTs, got {n_active}"

        before = {f: getattr(bank, f'mem_{f}').clone()
                  for f in ['keys', 'vals', 'importance', 'confidence', 'state']}

        # Write a very different vector (also low cosine to all existing)
        torch.manual_seed(999)
        h_new = torch.randn(1, 8)
        idx = bank.write(h_new, torch.zeros(1), torch.ones(1))
        target = idx[0].item()

        for s in range(config.memory_capacity):
            if s == target:
                assert not torch.equal(before['keys'][s], bank.mem_keys[s]), \
                    f"Target slot {s} key must have changed"
            else:
                assert torch.equal(before['keys'][s], bank.mem_keys[s]), \
                    f"Non-target slot {s} key must NOT have changed"
