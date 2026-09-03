"""
tests/test_replacement.py – Precise replacement policy tests (PHASE 11).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, STATE_ACTIVE, STATE_DORMANT, STATE_EXPIRED


class TestControlledReplacement:
    """PHASE 11: Exact controlled slot replacement testing."""

    def test_exact_lowest_importance_replacement(self):
        """
        Controlled test:
        slot 0 imp = 0.1
        slot 1 imp = 0.5
        slot 2 imp = 0.9
        slot 3 imp = 0.7
        Write new fact -> slot 0 must be replaced, slots 1, 2, 3 must be strictly unchanged.
        """
        cap = 4
        config = TinyMemoryConfig(
            memory_capacity=cap, memory_dim=8, hidden_size=8,
            memory_write_threshold=1.1,  # Force INSERT branch (no cosine update)
        )
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())

        # Fill all 4 slots
        for _ in range(cap):
            h = torch.randn(1, config.hidden_size)
            bank.write(h, torch.ones(1), torch.tensor([2.0]))

        # Impose explicit controlled state
        bank.mem_state.fill_(STATE_ACTIVE)
        bank.mem_importance.copy_(torch.tensor([0.1, 0.5, 0.9, 0.7]))
        bank.mem_confidence.fill_(0.9)

        keys_before = bank.mem_keys.clone()
        vals_before = bank.mem_vals.clone()
        imp_before = bank.mem_importance.clone()
        conf_before = bank.mem_confidence.clone()
        created_before = bank.mem_created_at.clone()
        last_acc_before = bank.mem_last_access.clone()
        acc_cnt_before = bank.mem_access_count.clone()

        # Step time forward
        bank.global_step[0] = 10

        # Write new fact
        h_new = torch.randn(1, config.hidden_size)
        bank.write(h_new, torch.ones(1), torch.tensor([2.0]))

        diff_keys = torch.sum((keys_before - bank.mem_keys) ** 2, dim=1)
        diff_vals = torch.sum((vals_before - bank.mem_vals) ** 2, dim=1)

        # Slot 0 MUST change completely
        assert diff_keys[0] > 1e-5, "Slot 0 key must be updated"
        assert diff_vals[0] > 1e-5, "Slot 0 val must be updated"
        assert bank.mem_created_at[0] == 10, "Slot 0 created_at must update to current step"
        assert bank.mem_last_access[0] == 10, "Slot 0 last_access must update to current step"
        assert bank.mem_confidence[0] == 0.5, "Slot 0 confidence must reset to new memory default"

        # Slots 1, 2, 3 MUST be strictly unchanged
        assert torch.all(diff_keys[1:] < 1e-5), "Slots 1, 2, 3 keys must remain unchanged"
        assert torch.all(diff_vals[1:] < 1e-5), "Slots 1, 2, 3 vals must remain unchanged"
        assert torch.all(imp_before[1:] == bank.mem_importance[1:]), "Slots 1, 2, 3 importance unchanged"
        assert torch.all(conf_before[1:] == bank.mem_confidence[1:]), "Slots 1, 2, 3 confidence unchanged"
        assert torch.all(created_before[1:] == bank.mem_created_at[1:]), "Slots 1, 2, 3 created_at unchanged"
        assert torch.all(last_acc_before[1:] == bank.mem_last_access[1:]), "Slots 1, 2, 3 last_acc unchanged"
        assert torch.all(acc_cnt_before[1:] == bank.mem_access_count[1:]), "Slots 1, 2, 3 acc_cnt unchanged"

    def test_replacement_prefers_expired_over_dormant_and_active(self):
        """Priority order: EXPIRED (0) < DORMANT (1) < ACTIVE (2)."""
        cap = 3
        config = TinyMemoryConfig(memory_capacity=cap, memory_dim=8, hidden_size=8, memory_write_threshold=1.1)
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())

        # Set slot 0 = ACTIVE, slot 1 = DORMANT, slot 2 = EXPIRED
        bank.mem_state[0] = STATE_ACTIVE
        bank.mem_state[1] = STATE_DORMANT
        bank.mem_state[2] = STATE_EXPIRED

        h = torch.randn(1, config.hidden_size)
        idx = bank.write(h, torch.ones(1), torch.tensor([2.0]))
        assert int(idx[0].item()) == 2, "Must choose EXPIRED slot 2 over DORMANT or ACTIVE"

        # Now slot 2 is ACTIVE. Next write should choose DORMANT slot 1
        h2 = torch.randn(1, config.hidden_size)
        idx2 = bank.write(h2, torch.ones(1), torch.tensor([2.0]))
        assert int(idx2[0].item()) == 1, "Must choose DORMANT slot 1 over ACTIVE slot 0"
