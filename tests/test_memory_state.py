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
        bank, config = small_bank
        torch.manual_seed(6)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h, wp=torch.full((1,), -1.0))
        assert int(torch.sum(bank.mem_state == STATE_ACTIVE)) == 0

    def test_write_gating_is_eos_zero(self, small_bank):
        bank, config = small_bank
        torch.manual_seed(7)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h, eos=torch.zeros(1))
        assert int(torch.sum(bank.mem_state == STATE_ACTIVE)) == 0

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
        config = TinyMemoryConfig(
            memory_capacity=4, memory_dim=4, hidden_size=4,
            memory_write_threshold=1.1,  # Force insert instead of update
        )
        bank = init_bank(config, seed=0)

        for i in range(4):
            torch.manual_seed(50 + i)
            h = torch.randn(1, 4)
            apply_write(bank, h)

        bank.mem_importance.copy_(torch.tensor([0.9, 0.5, 0.1, 0.7]))
        bank.mem_state.fill_(STATE_ACTIVE)

        torch.manual_seed(200)
        h_new = torch.randn(1, 4)
        apply_write(bank, h_new, wp=torch.tensor([2.0]))
        
        # Slot with lowest importance (index 2) should be replaced
        assert int(bank.mem_access_count[2]) == 1, "Slot with lowest importance was not replaced"
        assert int(bank.mem_access_count[0]) != 1, "Slot with high importance was replaced"
