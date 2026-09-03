"""
tests/test_counterfactual.py – PyTorch Counterfactual Causal Test
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig
from tests.conftest import init_bank, apply_write, apply_read


def cosine_sim(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a / na, b / nb))


class TestCounterfactualCausal:
    @pytest.fixture
    def setup(self):
        config = TinyMemoryConfig(
            memory_capacity=16, memory_dim=8, hidden_size=8,
            memory_top_k=4,
            mem_decay_rate=0.0001,
        )
        bank = init_bank(config, seed=0)
        return bank, config

    def test_same_key_same_value_high_similarity(self, setup):
        """Same key + same params → identical retrieval."""
        bank, config = setup
        torch.manual_seed(43)
        K = torch.randn(1, config.hidden_size)

        bank.load_memory_state(bank.empty_memory_state())
        apply_write(bank, K)
        RA1 = apply_read(bank, K)

        bank.load_memory_state(bank.empty_memory_state())
        apply_write(bank, K)
        RA2 = apply_read(bank, K)

        assert torch.allclose(RA1, RA2, atol=1e-5), "Same key + same value → identical retrievals"

    def test_true_causality_of_stored_value(self, setup):
        """Manually inject different value into slot → different read output. Proves true causality."""
        bank, config = setup
        torch.manual_seed(44)
        K = torch.randn(1, config.hidden_size)

        bank.load_memory_state(bank.empty_memory_state())
        apply_write(bank, K)
        R_original = apply_read(bank, K).clone()

        # Inject negated value in active slot
        active_slots = torch.where(bank.mem_state == 1)[0]
        assert len(active_slots) > 0
        idx = int(active_slots[0])
        bank.mem_vals[idx] = -bank.mem_vals[idx]

        R_modified = apply_read(bank, K)
        assert not torch.allclose(R_original, R_modified, atol=1e-4), \
            "Changing stored value must change read output"

        sim = cosine_sim(R_original[0].detach().numpy(), R_modified[0].detach().numpy())
        assert sim < 0.5, f"Negated value should produce different retrieval. sim={sim:.4f}"

    def test_minimally_perturbed_counterfactual(self, setup):
        """P1 Counterfactual: Distinguish Fact A from minimally perturbed Fact A'."""
        bank, config = setup
        dim = config.memory_dim

        # Fact A
        torch.manual_seed(999)
        fact_A = torch.randn(1, dim)
        fact_A = fact_A / torch.norm(fact_A)

        # Fact A' (slightly perturbed)
        noise = torch.randn(1, dim)
        fact_A_prime = fact_A + 0.1 * noise
        fact_A_prime = fact_A_prime / torch.norm(fact_A_prime)

        bank.load_memory_state(bank.empty_memory_state())
        bank.config.memory_write_threshold = 1.05  # Force insertion into distinct slots
        slot_A = bank.write(fact_A, torch.ones(1), torch.tensor([2.0]))[0].item()
        slot_A_prime = bank.write(fact_A_prime, torch.ones(1), torch.tensor([2.0]))[0].item()

        assert slot_A != slot_A_prime, f"Fact A (slot {slot_A}) and Fact A' (slot {slot_A_prime}) must occupy distinct slots"

        # Read using Fact A query
        _, scores = bank.read(fact_A, return_scores=True)
        scores_np = scores[0].detach().cpu().numpy()

        # Both slots should have valid scores
        assert scores_np[slot_A] > -1e8
        assert scores_np[slot_A_prime] > -1e8
        # Fact A and Fact A' must produce differentiated scores
        assert abs(scores_np[slot_A] - scores_np[slot_A_prime]) > 1e-5
