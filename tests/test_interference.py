"""
tests/test_interference.py – Interference and distractor resistance tests.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig
from dataset.generator import generate_random_dataset, generate_interference_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity


class TestInterferenceResistance:
    @pytest.fixture
    def setup(self):
        config = TinyMemoryConfig(memory_capacity=64, memory_dim=16, hidden_size=16, memory_top_k=4)
        torch.manual_seed(123)
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())
        return bank, config

    def test_low_noise_interference(self, setup):
        """Distractors with low noise (0.1) should not destroy target retrieval."""
        bank, config = setup
        dim = config.memory_dim

        target = generate_random_dataset(42, 1, dim)
        t_eos, t_is_eos, t_w_prob, _ = create_synthetic_batch(target)
        bank.write(t_eos, t_is_eos, t_w_prob)

        distractors = generate_interference_dataset(43, target, 10, noise_level=0.1)
        d_eos, d_is_eos, d_w_prob, _ = create_synthetic_batch(distractors)
        for i in range(10):
            bank.write(d_eos[i:i+1], d_is_eos[i:i+1], d_w_prob[i:i+1])

        retrieved_v = bank.read(t_eos)
        expected_v = bank.v_proj(t_eos)

        sim = compute_cosine_similarity(retrieved_v.detach().numpy(), expected_v.detach().numpy())
        assert sim[0] > 0.5, f"Low noise interference similarity too low: {sim[0]:.4f}"

    def test_orthogonal_distractors_retrieval_integrity(self, setup):
        """Multiple completely orthogonal distractors must not prevent target retrieval."""
        bank, config = setup
        dim = config.hidden_size

        target = torch.randn(1, dim)
        target = target / torch.norm(target)
        bank.write(target, torch.ones(1), torch.ones(1))

        for _ in range(15):
            d = torch.randn(1, dim)
            d = d / torch.norm(d)
            bank.write(d, torch.ones(1), torch.ones(1))

        # Target slot is 0
        _, scores = bank.read(target, return_scores=True)
        top_slot = torch.argmax(scores[0]).item()
        assert top_slot == 0, f"Target slot 0 should have highest score, got slot {top_slot}"
