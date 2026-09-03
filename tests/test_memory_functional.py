"""
tests/test_memory_functional.py – PyTorch Functional Memory Bank Tests
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, STATE_ACTIVE
from tests.conftest import init_bank, apply_write, apply_read


def cosine_float(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a / na, b / nb))


@pytest.fixture
def bank_factory():
    def _make(capacity=32, dim=16, hidden=16, top_k=4):
        config = TinyMemoryConfig(
            memory_capacity=capacity, memory_dim=dim, hidden_size=hidden,
            memory_top_k=top_k,
            mem_decay_rate=0.0001,
            memory_write_threshold=0.9,
        )
        bank = init_bank(config, seed=0)
        
        # Inject identity matrices for q_proj and k_proj
        with torch.no_grad():
            bank.q_proj.weight.data.copy_(torch.eye(dim, hidden))
            bank.k_proj.weight.data.copy_(torch.eye(dim, hidden))
        
        return bank, config
    return _make


class TestBasicWriteRead:
    def test_write_one_fact_active_count(self, bank_factory):
        bank, config = bank_factory()
        torch.manual_seed(1)
        h = torch.randn(1, config.hidden_size)
        apply_write(bank, h)
        assert int(torch.sum(bank.mem_state == STATE_ACTIVE)) == 1

    def test_write_read_cosine_similarity(self, bank_factory):
        bank, config = bank_factory()
        torch.manual_seed(2)
        h = torch.randn(1, config.hidden_size)
        h = h / torch.norm(h)

        apply_write(bank, h)
        out = apply_read(bank, h)

        expected_v = bank.v_proj(h)
        sim = cosine_float(out[0].detach().numpy(), expected_v[0].detach().numpy())
        assert sim > 0.0, f"Write-Read cosine similarity too low: {sim:.4f}"


class TestDistractorRetrieval:
    def test_recall_at_1_with_distractors(self, bank_factory):
        bank, config = bank_factory(capacity=64, top_k=8)
        torch.manual_seed(10)

        h_target = torch.randn(1, config.hidden_size)
        h_target = h_target / torch.norm(h_target)
        apply_write(bank, h_target)

        for _ in range(10):
            h_d = torch.randn(1, config.hidden_size)
            h_d = h_d / torch.norm(h_d)
            apply_write(bank, h_d)

        out = apply_read(bank, h_target)
        expected_v = bank.v_proj(h_target)
        sim = cosine_float(out[0].detach().numpy(), expected_v[0].detach().numpy())
        assert sim > 0.7, f"Target should be cleanly retrieved despite distractors. Sim={sim:.4f}"


class TestEmptySlotMasking:
    def test_one_active_top_k_eight(self, bank_factory):
        bank, config = bank_factory(capacity=16, top_k=8)
        h = torch.ones(1, config.hidden_size) * 3.0
        apply_write(bank, h)
        out = apply_read(bank, h)
        assert torch.norm(out) > 1e-6

    def test_zero_active_returns_zero(self, bank_factory):
        bank, config = bank_factory()
        h = torch.ones(1, config.hidden_size)
        out = apply_read(bank, h)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


class TestInterference:
    def test_target_retrievable_after_interference(self, bank_factory):
        bank, config = bank_factory(capacity=64, top_k=8)
        torch.manual_seed(100)

        h_target = torch.randn(1, config.hidden_size)
        h_target = h_target / torch.norm(h_target)
        apply_write(bank, h_target)

        for _ in range(20):
            h_d = torch.randn(1, config.hidden_size)
            apply_write(bank, h_d)

        out = apply_read(bank, h_target)
        expected_v = bank.v_proj(h_target)
        sim = cosine_float(out[0].detach().numpy(), expected_v[0].detach().numpy())
        assert sim > 0.3, f"Target should survive interference. Sim={sim:.4f}"


class TestCapacityScaling:
    @pytest.mark.parametrize("capacity", [16, 32, 64, 128])
    def test_capacity_allows_more_storage(self, bank_factory, capacity):
        bank, config = bank_factory(capacity=capacity, top_k=4)
        torch.manual_seed(200)

        n_writes = min(capacity, 30)
        for _ in range(n_writes):
            h = torch.randn(1, config.hidden_size)
            apply_write(bank, h)

        active = int(torch.sum(bank.mem_state == STATE_ACTIVE))
        assert active >= int(n_writes * 0.8), f"Expected ~{n_writes} active with cap={capacity}, got {active}"
