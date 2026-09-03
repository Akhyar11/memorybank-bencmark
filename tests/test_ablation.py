"""
tests/test_ablation.py – Unit tests for component ablations.
Verifies that disabling exactly one mechanism removes only its intended effect.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, STATE_ACTIVE
from experiments.ablation import make_ablation_config


class TestComponentAblations:
    @pytest.fixture
    def base_setup(self):
        config = make_ablation_config('none')
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())
        return bank, config

    def test_no_write_blocks_all_modifications(self):
        """PHASE 10: When WRITE is disabled, all memory fields must remain strictly unchanged."""
        config = make_ablation_config('no_write')
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())

        mem_before = {k: v.clone() for k, v in bank.get_memory_state().items()}

        h = torch.randn(2, config.hidden_size)
        bank.write(h, torch.ones(2), torch.ones(2))

        mem_after = bank.get_memory_state()
        for k in mem_before:
            assert torch.equal(mem_before[k], mem_after[k]), f"no_write altered field {k}"

        active_count = int(torch.sum(bank.mem_state == STATE_ACTIVE))
        assert active_count == 0, "No active memory slots should exist after blocked write"

    def test_no_read_produces_zero_and_preserves_state(self):
        """PHASE 10: When READ is disabled, output is zero and access metadata is unchanged."""
        config = make_ablation_config('no_read')
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())

        # First write one memory with write_threshold allowing write
        with torch.no_grad():
            bank.config.memory_write_threshold = 0.0
            h = torch.randn(1, config.hidden_size)
            bank.write(h, torch.ones(1), torch.ones(1))
            bank.config.memory_write_threshold = 1e9

        mem_before = {k: v.clone() for k, v in bank.get_memory_state().items()}

        out = bank.read(h, read_prob=torch.ones(1))
        assert float(torch.norm(out)) == 0.0, "no_read must produce strictly zero vector"

        mem_after = bank.get_memory_state()
        for k in ['last_access', 'access_count', 'importance']:
            assert torch.equal(mem_before[k], mem_after[k]), f"no_read altered access metadata {k}"

    def test_no_decay_preserves_active_state_forever(self):
        """When decay is disabled (rate=0), memory never expires."""
        config = make_ablation_config('no_decay')
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())

        h = torch.randn(1, config.hidden_size)
        bank.write(h, torch.ones(1), torch.ones(1))

        # Advance huge amount of time
        bank.global_step[0] = 1_000_000
        bank.decay_memory()

        assert bank.mem_state[0] == STATE_ACTIVE, "no_decay must preserve ACTIVE state"

    def test_no_reinforcement_does_not_boost_importance(self):
        """When reinforcement rate is 0, reading does not increment importance."""
        config = make_ablation_config('no_reinforcement')
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())

        h = torch.randn(1, config.hidden_size)
        bank.write(h, torch.ones(1), torch.ones(1))
        initial_importance = bank.mem_importance[0].item()

        for _ in range(5):
            bank.read(h)

        final_importance = bank.mem_importance[0].item()
        assert initial_importance == final_importance, "no_reinforcement should not boost importance"

    def test_no_recency_ignores_time_delay(self):
        """When mem_gamma is 0.0, time step dt does not influence retrieval scores."""
        config = make_ablation_config('no_recency')
        config.mem_reinforcement_rate = 0.0  # Isolate from access reinforcement boost
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())

        dim = config.memory_dim
        h = torch.randn(1, dim)
        bank.write(h, torch.ones(1), torch.ones(1))

        # Score at step 0
        _, scores_step0 = bank.read(h, return_scores=True)

        # Advance step to 1000 (without decay)
        bank.global_step[0] = 1000
        _, scores_step1000 = bank.read(h, return_scores=True)

        # In no_recency, temporal recency term gamma * recency is 0, so scores remain identical
        assert torch.allclose(scores_step0, scores_step1000, atol=1e-5), "no_recency should ignore time progression in scores"

    def test_no_importance_ignores_slot_importance(self):
        """When mem_beta is 0.0, slot importance does not alter score ranking."""
        config = make_ablation_config('no_importance')
        config.mem_reinforcement_rate = 0.0
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())

        dim = config.memory_dim
        h = torch.randn(1, dim)
        bank.write(h, torch.ones(1), torch.ones(1))

        _, score_before = bank.read(h, return_scores=True)

        # Modify importance manually
        bank.mem_importance[0] = 1.0
        _, score_after = bank.read(h, return_scores=True)

        assert torch.allclose(score_before, score_after, atol=1e-5), "no_importance must not reflect importance changes in score"

    def test_no_confidence_ignores_confidence_metadata(self):
        """When mem_delta is 0.0, confidence changes do not alter retrieval scores."""
        config = make_ablation_config('no_confidence')
        config.mem_reinforcement_rate = 0.0  # Isolate from access reinforcement boost
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())

        dim = config.memory_dim
        h = torch.randn(1, dim)
        bank.write(h, torch.ones(1), torch.ones(1))

        _, score_before = bank.read(h, return_scores=True)

        # Modify confidence manually
        bank.mem_confidence[0] = 0.99
        _, score_after = bank.read(h, return_scores=True)

        assert torch.allclose(score_before, score_after, atol=1e-5), "no_confidence must not reflect confidence changes in score"
