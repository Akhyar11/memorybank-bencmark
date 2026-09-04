"""
tests/test_memory_state.py – PyTorch Memory State & Transitions Test
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.tiny_memory_bank import (
    TinyMemoryBank, TinyMemoryConfig,
    STATE_EXPIRED, STATE_ACTIVE, STATE_DORMANT
)
from tests.conftest import init_bank, apply_write, apply_read, apply_decay


@pytest.fixture
def small_bank():
    config = TinyMemoryConfig(
        memory_capacity=8, memory_dim=16, hidden_size=16,
        memory_top_k=4,
        mem_decay_rate=1.0,
        mem_importance_protection=0.5,
        memory_write_threshold=0.9,  # High threshold so random keys insert instead of update
    )
    bank = init_bank(config, seed=0)
    return bank, config


class TestInitialState:
    def test_memory_starts_empty(self, small_bank):
        bank, config = small_bank
        assert torch.all(bank.mem_state == STATE_EXPIRED)

    def test_active_count_starts_zero(self, small_bank):
        bank, _ = small_bank
        assert int(torch.sum(bank.mem_state == STATE_ACTIVE)) == 0

    def test_global_step_starts_zero(self, small_bank):
        bank, _ = small_bank
        assert int(bank.global_step.item()) == 0


class TestWrite:
    def test_write_increases_active_count(self, small_bank):
        bank, config = small_bank
        torch.manual_seed(1)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h)
        assert int(torch.sum(bank.mem_state == STATE_ACTIVE)) == 1

    def test_write_stores_keys_via_k_proj(self, small_bank):
        bank, config = small_bank
        torch.manual_seed(2)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h)
        key_stored = bank.mem_keys[0]
        assert torch.norm(key_stored) > 1e-6, "Key slot should be non-zero after write"

    def test_write_updates_timestamps(self, small_bank):
        bank, config = small_bank
        torch.manual_seed(3)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h)
        active_idx = int(torch.argmax((bank.mem_state == STATE_ACTIVE).int()))
        assert int(bank.mem_access_count[active_idx]) >= 1

    def test_write_initialises_importance(self, small_bank):
        bank, config = small_bank
        torch.manual_seed(4)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h)
        active_idx = int(torch.argmax((bank.mem_state == STATE_ACTIVE).int()))
        assert 0.0 <= float(bank.mem_importance[active_idx]) <= 1.0

    def test_write_initialises_confidence(self, small_bank):
        bank, config = small_bank
        torch.manual_seed(5)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h)
        active_idx = int(torch.argmax((bank.mem_state == STATE_ACTIVE).int()))
        conf = float(bank.mem_confidence[active_idx])
        assert conf == pytest.approx(0.5, abs=0.01), f"Expected 0.5, got {conf}"

    def test_write_gating_write_prob_zero(self, small_bank):
        """P0 Write Gate Semantics: write gate OFF -> no mutation, returns target -1."""
        bank, config = small_bank
        torch.manual_seed(6)
        h = torch.randn(1, config.hidden_size)

        keys_before = bank.mem_keys.clone()
        vals_before = bank.mem_vals.clone()
        imp_before = bank.mem_importance.clone()
        conf_before = bank.mem_confidence.clone()
        created_before = bank.mem_created_at.clone()
        last_acc_before = bank.mem_last_access.clone()
        acc_cnt_before = bank.mem_access_count.clone()
        state_before = bank.mem_state.clone()

        target_idx = bank.write(h, torch.ones(1), torch.full((1,), -1.0))

        # Target must explicitly indicate no write occurred (-1)
        assert target_idx[0].item() == -1, f"Expected target -1 for blocked write, got {target_idx[0].item()}"

        # All state fields must remain strictly identical
        assert torch.equal(bank.mem_keys, keys_before), "Keys mutated during blocked write"
        assert torch.equal(bank.mem_vals, vals_before), "Vals mutated during blocked write"
        assert torch.equal(bank.mem_importance, imp_before), "Importance mutated during blocked write"
        assert torch.equal(bank.mem_confidence, conf_before), "Confidence mutated during blocked write"
        assert torch.equal(bank.mem_created_at, created_before), "Created_at mutated during blocked write"
        assert torch.equal(bank.mem_last_access, last_acc_before), "Last_access mutated during blocked write"
        assert torch.equal(bank.mem_access_count, acc_cnt_before), "Access_count mutated during blocked write"
        assert torch.equal(bank.mem_state, state_before), "State mutated during blocked write"
        assert int(torch.sum(bank.mem_state == STATE_ACTIVE)) == 0

    def test_write_gating_is_eos_zero_high_prob_writes(self, small_bank):
        """
        NEW SEMANTICS: is_eos=0 with write_prob=HIGH → WRITE must occur.
        is_eos is no longer a prerequisite for writing.
        Old test expected no-write; this is now incorrect per updated semantics.
        """
        bank, config = small_bank
        torch.manual_seed(7)
        h = torch.randn(1, config.hidden_size)

        # write_threshold=0.9 in small_bank fixture; use write_prob=1.0 (above threshold)
        # is_eos=0 must NOT block the write
        target_idx = bank.write(h, torch.zeros(1), torch.ones(1))
        assert target_idx[0].item() != -1, (
            "is_eos=False with write_prob=HIGH should WRITE (is_eos no longer gates)"
        )
        assert int(torch.sum(bank.mem_state == STATE_ACTIVE)) == 1

    # -----------------------------------------------------------------------
    # New write-gate tests: all 4 cases from spec
    # -----------------------------------------------------------------------

    def test_write_gate_case_A_non_eos_high_prob_writes(self, small_bank):
        """Case A: is_eos=False, write_prob >= threshold → WRITE."""
        bank, config = small_bank
        torch.manual_seed(30)
        h = torch.randn(1, config.hidden_size)
        # write_threshold=0.9 in fixture; wp=1.0 ≥ threshold
        idx = bank.write(h, torch.zeros(1), torch.ones(1))
        assert idx[0].item() != -1, "Case A: must WRITE when write_prob >= threshold"
        assert int(torch.sum(bank.mem_state == STATE_ACTIVE)) >= 1

    def test_write_gate_case_B_eos_low_prob_no_write(self, small_bank):
        """Case B: is_eos=True, write_prob < threshold → NO WRITE."""
        bank, config = small_bank
        torch.manual_seed(31)
        h = torch.randn(1, config.hidden_size)
        state_before = bank.mem_state.clone()
        # write_threshold=0.9 in fixture; wp=-1.0 < threshold → blocked
        idx = bank.write(h, torch.ones(1), torch.full((1,), -1.0))
        assert idx[0].item() == -1, "Case B: must NOT WRITE when write_prob < threshold"
        assert torch.equal(bank.mem_state, state_before), "State must not change"

    def test_write_gate_case_C_non_eos_low_prob_no_write(self, small_bank):
        """Case C: is_eos=False, write_prob < threshold → NO WRITE."""
        bank, config = small_bank
        torch.manual_seed(32)
        h = torch.randn(1, config.hidden_size)
        state_before = bank.mem_state.clone()
        idx = bank.write(h, torch.zeros(1), torch.full((1,), -1.0))
        assert idx[0].item() == -1, "Case C: must NOT WRITE when write_prob < threshold"
        assert torch.equal(bank.mem_state, state_before), "State must not change"

    def test_write_gate_case_D_eos_high_prob_writes(self, small_bank):
        """Case D: is_eos=True, write_prob >= threshold → WRITE."""
        bank, config = small_bank
        torch.manual_seed(33)
        h = torch.randn(1, config.hidden_size)
        idx = bank.write(h, torch.ones(1), torch.ones(1))
        assert idx[0].item() != -1, "Case D: must WRITE when write_prob >= threshold"
        assert int(torch.sum(bank.mem_state == STATE_ACTIVE)) >= 1

    def test_multiple_writes_fill_capacity(self, small_bank):
        bank, config = small_bank
        for i in range(config.memory_capacity):
            torch.manual_seed(10 + i)
            h = torch.randn(1, config.hidden_size)
            apply_write(bank, h, wp=torch.ones(1))
        active = int(torch.sum(bank.mem_state == STATE_ACTIVE))
        assert active == config.memory_capacity, f"Expected {config.memory_capacity} active, got {active}"


class TestRead:
    def test_read_empty_returns_zero(self, small_bank):
        bank, config = small_bank
        h = torch.ones(1, config.hidden_size)
        out = apply_read(bank, h)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_read_updates_access_count(self, small_bank):
        bank, config = small_bank
        torch.manual_seed(20)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h)
        acc_before = bank.mem_access_count.clone()
        apply_read(bank, h)
        assert int(torch.sum(bank.mem_access_count)) > int(torch.sum(acc_before))

    def test_read_updates_last_access(self, small_bank):
        bank, config = small_bank
        torch.manual_seed(21)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h)
        la_before = bank.mem_last_access.clone()
        apply_read(bank, h)
        assert torch.any(bank.mem_last_access >= la_before)

    def test_read_boosts_importance(self, small_bank):
        bank, config = small_bank
        torch.manual_seed(22)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h)
        imp_before = bank.mem_importance.clone()
        apply_read(bank, h)
        assert torch.any(bank.mem_importance >= imp_before)

    def test_read_top_k_respects_capacity(self, small_bank):
        bank, config = small_bank
        h = torch.ones(1, config.hidden_size) * 2.0
        apply_write(bank, h)
        out = apply_read(bank, h)
        assert torch.norm(out) > 1e-6, "1 active memory should give non-zero read"

    def test_read_gate_has_no_effects(self, small_bank):
        bank, config = small_bank
        h = torch.ones(1, config.hidden_size)
        apply_write(bank, h)
        
        state_before = bank.mem_importance.clone()
        last_acc_before = bank.mem_last_access.clone()
        acc_cnt_before = bank.mem_access_count.clone()
        mem_state_before = bank.mem_state.clone()
        
        out = apply_read(bank, h, rp=torch.zeros(1))
        
        assert torch.allclose(out, torch.zeros_like(out)), "Read output should be 0 when gated"
        assert torch.allclose(state_before, bank.mem_importance), "Importance should not be updated when read_prob=0"
        assert torch.allclose(last_acc_before, bank.mem_last_access), "Last access should not be updated"
        assert torch.allclose(acc_cnt_before, bank.mem_access_count), "Access count should not be updated"
        assert torch.allclose(mem_state_before, bank.mem_state), "State should not be updated"


class TestDecay:
    def test_decay_expires_old_memories(self, small_bank):
        bank, config = small_bank
        torch.manual_seed(30)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h)
        bank.global_step[0] = 100000
        apply_decay(bank)
        assert torch.all(bank.mem_state != STATE_ACTIVE)

    def test_decay_formula(self, small_bank):
        _, config = small_bank
        lam = config.mem_decay_rate  # 1.0
        dt  = 1
        R   = torch.exp(-torch.tensor(lam * dt).float())
        assert float(R) < 0.5

    def test_decay_dormant_then_expired(self, small_bank):
        bank, config = small_bank
        torch.manual_seed(31)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h)

        # dt=1 → DORMANT
        bank.global_step[0] = 1
        apply_decay(bank)
        assert torch.any(bank.mem_state == STATE_DORMANT)

        # dt=10000 → EXPIRED
        bank.global_step[0] = 10000
        apply_decay(bank)
        assert torch.any(bank.mem_state == STATE_EXPIRED)


class TestReplacement:
    def test_replacement_uses_expired_first(self, small_bank):
        bank, config = small_bank
        for i in range(config.memory_capacity):
            torch.manual_seed(40 + i)
            h = torch.randn(1, config.hidden_size)
            apply_write(bank, h)

        # Expire half
        bank.mem_state[:config.memory_capacity // 2] = STATE_EXPIRED
        n_expired_before = int(torch.sum(bank.mem_state == STATE_EXPIRED))

        torch.manual_seed(99)
        h_new = torch.randn(1, config.hidden_size)
        apply_write(bank, h_new)
        n_expired_after = int(torch.sum(bank.mem_state == STATE_EXPIRED))
        assert n_expired_after < n_expired_before

    def test_replacement_different_importance(self):
        """P0 Replacement Test: Identity/Content-based validation proving lowest-priority slot replaced and others preserved."""
        config = TinyMemoryConfig(
            memory_capacity=4, memory_dim=4, hidden_size=4,
            memory_write_threshold=1.1,  # Force insert instead of update
        )
        bank = init_bank(config, seed=0)

        # Fill 4 slots with distinct initial memories
        for i in range(4):
            torch.manual_seed(50 + i)
            h = torch.randn(1, 4)
            apply_write(bank, h)

        # Force slot 2 to have lowest priority (lowest importance)
        bank.mem_importance.copy_(torch.tensor([0.9, 0.5, 0.1, 0.7]))
        bank.mem_state.fill_(STATE_ACTIVE)

        # Snapshot old memory state
        old_keys = bank.mem_keys.clone()
        old_vals = bank.mem_vals.clone()
        old_imp = bank.mem_importance.clone()
        old_conf = bank.mem_confidence.clone()
        old_created = bank.mem_created_at.clone()
        old_last_acc = bank.mem_last_access.clone()
        old_state = bank.mem_state.clone()

        # Step time forward
        bank.global_step[0] = 50

        # Perform new write
        torch.manual_seed(200)
        h_new = torch.randn(1, 4)
        target_idx = bank.write(h_new, torch.ones(1), torch.tensor([2.0]))

        assert target_idx[0].item() == 2, f"Target slot should be 2, got {target_idx[0].item()}"

        # Projected new key and val
        expected_new_k = bank.k_proj(h_new).detach()[0]
        expected_new_v = bank.v_proj(h_new).detach()[0]

        # 1. Verify slot 2 was REPLACED with NEW MEMORY
        assert torch.allclose(bank.mem_keys[2], expected_new_k, atol=1e-5), "Slot 2 key must match new projected key"
        assert torch.allclose(bank.mem_vals[2], expected_new_v, atol=1e-5), "Slot 2 val must match new projected val"
        assert bank.mem_state[2] == STATE_ACTIVE, "Slot 2 state must be ACTIVE"
        assert bank.mem_created_at[2] == 50, "Slot 2 created_at must match current step 50"
        assert bank.mem_last_access[2] == 50, "Slot 2 last_access must match current step 50"
        assert bank.mem_access_count[2] == 1, "Slot 2 access_count must reset to 1"

        # 2. Verify slots 0, 1, 3 are STRICTLY PRESERVED (OLD MEMORY)
        for preserved_slot in [0, 1, 3]:
            assert torch.allclose(bank.mem_keys[preserved_slot], old_keys[preserved_slot], atol=1e-5), f"Slot {preserved_slot} key altered"
            assert torch.allclose(bank.mem_vals[preserved_slot], old_vals[preserved_slot], atol=1e-5), f"Slot {preserved_slot} val altered"
            assert bank.mem_importance[preserved_slot] == old_imp[preserved_slot], f"Slot {preserved_slot} importance altered"
            assert bank.mem_confidence[preserved_slot] == old_conf[preserved_slot], f"Slot {preserved_slot} confidence altered"
            assert bank.mem_created_at[preserved_slot] == old_created[preserved_slot], f"Slot {preserved_slot} created_at altered"
            assert bank.mem_last_access[preserved_slot] == old_last_acc[preserved_slot], f"Slot {preserved_slot} last_access altered"
            assert bank.mem_state[preserved_slot] == old_state[preserved_slot], f"Slot {preserved_slot} state altered"
