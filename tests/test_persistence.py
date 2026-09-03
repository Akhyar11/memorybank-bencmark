"""
tests/test_persistence.py – Memory persistence and temporal clock tests (PHASE 19, 20).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, STATE_ACTIVE, STATE_DORMANT, STATE_EXPIRED


class TestMemoryPersistence:
    @pytest.fixture
    def bank_fixture(self):
        config = TinyMemoryConfig(
            memory_capacity=32, memory_dim=16, hidden_size=16,
            memory_top_k=4, mem_decay_rate=0.01,
        )
        torch.manual_seed(55)
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())
        return bank, config

    def test_persistent_memory_across_interactions(self, bank_fixture):
        """PHASE 19: Write fact -> multiple unrelated interactions -> retrieve fact without reset."""
        bank, config = bank_fixture

        # Step 1: Write target fact
        h_target = torch.randn(1, config.hidden_size)
        h_target = h_target / torch.norm(h_target)
        bank.write(h_target, torch.ones(1), torch.ones(1))

        # Step 2: Next interactions (several unrelated reads/writes)
        for _ in range(5):
            dummy = torch.randn(1, config.hidden_size)
            bank.read(dummy)

        # Step 3: Target fact must still be retrievable
        out = bank.read(h_target)
        expected_v = bank.v_proj(h_target)
        sim = torch.cosine_similarity(out[0], expected_v[0], dim=0).item()
        assert sim > 0.5, f"Persistent memory retrieval degraded: sim={sim:.4f}"

    def test_global_step_temporal_clock(self, bank_fixture):
        """PHASE 20: global_step functions as a coherent chronological clock."""
        bank, config = bank_fixture
        assert bank.global_step[0] == 0

        # Step 0: Write
        h = torch.randn(1, config.hidden_size)
        bank.write(h, torch.ones(1), torch.ones(1))
        assert bank.mem_created_at[0] == 0

        # Step 10: Read at step 10
        bank.global_step[0] = 10
        bank.read(h)
        assert bank.mem_last_access[0] == 10

        # Step 500: Decay at step 500 -> must transition to DORMANT or EXPIRED
        bank.global_step[0] = 500
        bank.decay_memory()
        assert bank.mem_state[0] in (STATE_DORMANT, STATE_EXPIRED)

    def test_state_save_reset_restore_exact_match(self, bank_fixture):
        """PHASE 2: save state -> reset -> restore produces 100% identical state."""
        bank, config = bank_fixture

        for _ in range(5):
            h = torch.randn(1, config.hidden_size)
            bank.write(h, torch.ones(1), torch.ones(1))

        saved_state = bank.get_memory_state()

        # Reset completely
        bank.load_memory_state(bank.empty_memory_state())
        assert torch.sum(bank.mem_state == STATE_ACTIVE) == 0

        # Restore
        bank.load_memory_state(saved_state)
        restored_state = bank.get_memory_state()

        for k in saved_state:
            assert torch.equal(saved_state[k], restored_state[k]), f"State mismatch on key {k}"
