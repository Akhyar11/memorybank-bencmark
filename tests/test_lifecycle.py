"""
tests/test_lifecycle.py – Full End-to-End Lifecycle Verification Test (P1).

Exercises the complete lifecycle sequence through actual method invocations:
WRITE -> READ -> REINFORCEMENT -> TIME ADVANCE -> DECAY -> READ -> REPLACEMENT
Validates that every state transition occurs naturally according to architectural rules.
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


class TestMemoryLifecycle:
    def test_complete_method_driven_lifecycle(self):
        """
        Executes actual method pipeline:
        1. WRITE fact 0 -> verify active slot and timestamps
        2. READ fact 0 -> verify access count increments and importance boosts (reinforcement)
        3. TIME ADVANCE & DECAY -> verify transition to DORMANT then EXPIRED
        4. WRITE to over-capacity -> verify REPLACEMENT reuses the EXPIRED slot
        """
        cap = 2
        dim = 8
        config = TinyMemoryConfig(
            memory_capacity=cap,
            memory_dim=dim,
            hidden_size=dim,
            memory_top_k=1,
            mem_decay_rate=0.05,
            mem_reinforcement_rate=0.1,
            memory_threshold=-1.0,
            memory_write_threshold=0.9
        )
        torch.manual_seed(100)
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())

        # Step 1: WRITE initial fact A
        h_A = torch.randn(1, dim)
        target_slot_A = bank.write(h_A, torch.ones(1), torch.ones(1))[0].item()
        assert target_slot_A == 0, f"Initial write should land in slot 0, got {target_slot_A}"
        assert bank.mem_state[0] == STATE_ACTIVE
        assert bank.mem_access_count[0] == 1
        initial_imp = bank.mem_importance[0].item()

        # Step 2: READ fact A -> triggers access reinforcement
        read_out = bank.read(h_A)
        assert torch.norm(read_out) > 0.0, "Read output should be non-zero"
        assert bank.mem_access_count[0] == 2, "Read must increment access count"
        reinforced_imp = bank.mem_importance[0].item()
        assert reinforced_imp > initial_imp, f"Reinforcement must increase importance: {reinforced_imp} vs {initial_imp}"

        # Step 3: WRITE fact B into remaining slot
        h_B = torch.randn(1, dim)
        target_slot_B = bank.write(h_B, torch.ones(1), torch.ones(1))[0].item()
        assert target_slot_B == 1, f"Second write should land in slot 1, got {target_slot_B}"
        assert bank.mem_state[1] == STATE_ACTIVE

        # Step 4: TIME ADVANCE -> Step clock to 50
        bank.global_step[0] = 50

        # Step 5: DECAY -> Both facts decay over 50 steps
        bank.decay_memory()
        # With lambda=0.05, dt=50 -> exp(-2.5) ~ 0.082 < 0.1 -> EXPIRED
        assert bank.mem_state[0] in (STATE_DORMANT, STATE_EXPIRED)
        assert bank.mem_state[1] in (STATE_DORMANT, STATE_EXPIRED)

        # Force further decay if dormant
        bank.global_step[0] = 500
        bank.decay_memory()
        assert bank.mem_state[0] == STATE_EXPIRED
        assert bank.mem_state[1] == STATE_EXPIRED

        # Step 6: WRITE fact C -> capacity full of EXPIRED slots
        # Replacement mechanism must automatically reuse slot 0 (first expired)
        h_C = torch.randn(1, dim)
        target_slot_C = bank.write(h_C, torch.ones(1), torch.ones(1))[0].item()
        assert target_slot_C in (0, 1), f"Replacement must select an expired slot, got {target_slot_C}"
        assert bank.mem_state[target_slot_C] == STATE_ACTIVE
        assert bank.mem_created_at[target_slot_C] == 500, "Created timestamp must update to step 500"
        assert bank.mem_access_count[target_slot_C] == 1, "Access count must reset to 1 for new insert"
