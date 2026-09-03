"""
tests/test_causal.py – Individual causal intervention & gradient semantics tests.
Verifies that each projection (q, k, v, i, fusion) causally alters downstream outputs,
and validates backward gradient availability on trainable parameters.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig


@pytest.fixture
def bank_fixture():
    config = TinyMemoryConfig(
        memory_capacity=16, memory_dim=8, hidden_size=8,
        memory_top_k=4, mem_decay_rate=0.0001,
    )
    torch.manual_seed(42)
    bank = TinyMemoryBank(config=config)
    bank.load_memory_state(bank.empty_memory_state())
    return bank, config


class TestCausalProjections:
    """PHASE 22: Individual causal intervention tests for each projection."""

    def test_causal_v_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.randn(1, config.hidden_size)

        bank.load_memory_state(bank.empty_memory_state())
        bank.write(h, torch.ones(1), torch.ones(1))
        out_orig = bank.read(h).clone()

        with torch.no_grad():
            bank.v_proj.weight.data.neg_()

        bank.load_memory_state(bank.empty_memory_state())
        bank.write(h, torch.ones(1), torch.ones(1))
        out_intervened = bank.read(h).clone()

        assert not torch.allclose(out_orig, out_intervened, atol=1e-4)
        assert torch.allclose(out_orig, -out_intervened, atol=1e-4)

    def test_causal_k_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.randn(1, config.hidden_size)

        bank.load_memory_state(bank.empty_memory_state())
        bank.write(h, torch.ones(1), torch.ones(1))
        key_orig = bank.mem_keys[0].clone()

        with torch.no_grad():
            bank.k_proj.weight.data.mul_(2.0)

        bank.load_memory_state(bank.empty_memory_state())
        bank.write(h, torch.ones(1), torch.ones(1))
        key_intervened = bank.mem_keys[0].clone()

        assert not torch.allclose(key_orig, key_intervened, atol=1e-4)

    def test_causal_q_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.randn(1, config.hidden_size)

        bank.load_memory_state(bank.empty_memory_state())
        bank.write(h, torch.ones(1), torch.ones(1))
        _, scores_orig = bank.read(h, return_scores=True)

        with torch.no_grad():
            bank.q_proj.weight.data.neg_()

        _, scores_intervened = bank.read(h, return_scores=True)
        assert not torch.allclose(scores_orig, scores_intervened, atol=1e-4)

    def test_causal_i_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.randn(1, config.hidden_size)

        bank.load_memory_state(bank.empty_memory_state())
        bank.write(h, torch.ones(1), torch.ones(1))
        imp_orig = bank.mem_importance[0].clone()

        with torch.no_grad():
            bank.i_proj.weight.data.neg_()

        bank.load_memory_state(bank.empty_memory_state())
        bank.write(h, torch.ones(1), torch.ones(1))
        imp_intervened = bank.mem_importance[0].clone()

        assert not torch.allclose(imp_orig, imp_intervened, atol=1e-4)

    def test_causal_fusion_proj(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.randn(1, config.hidden_size)

        bank.load_memory_state(bank.empty_memory_state())
        bank.write(h, torch.ones(1), torch.ones(1))
        fused_orig = bank(h, torch.ones(1), torch.zeros(1)).clone()

        with torch.no_grad():
            bank.fusion_proj.weight.data.neg_()

        fused_intervened = bank(h, torch.ones(1), torch.zeros(1)).clone()
        assert not torch.allclose(fused_orig, fused_intervened, atol=1e-4)
        assert torch.allclose(fused_orig, -fused_intervened, atol=1e-4)


class TestGradientSemantics:
    """PHASE 3: Verify gradient flow to trainable parameters."""

    def test_trainable_parameters_receive_gradients(self, bank_fixture):
        bank, config = bank_fixture
        h = torch.randn(2, config.hidden_size, requires_grad=True)

        # Write first
        bank.write(h, torch.ones(2), torch.ones(2))

        # Forward through memory
        fused = bank(h, torch.ones(2), torch.zeros(2))
        loss = torch.sum(fused ** 2)
        loss.backward()

        # Check gradient availability
        assert bank.q_proj.weight.grad is not None, "q_proj must receive gradients"
        assert bank.fusion_proj.weight.grad is not None, "fusion_proj must receive gradients"
        assert torch.norm(bank.q_proj.weight.grad) > 0.0
        assert torch.norm(bank.fusion_proj.weight.grad) > 0.0

        # Memory state buffers must NOT have requires_grad
        for name, buf in bank.named_buffers():
            assert not buf.requires_grad, f"Buffer {name} must not require grad"

    def test_write_state_mutation_is_nondifferentiable_and_preserves_external_cache_semantics(self, bank_fixture):
        """
        P1 Gradient Semantics:
        Memory state mutation is non-differentiable.
        Gradient flows through projection/computation paths (q_proj, fusion_proj, backbone)
        but not through persistent episodic buffer mutations (which mimic Flax variable collection 'memory').
        """
        bank, config = bank_fixture
        h = torch.randn(1, config.hidden_size, requires_grad=True)

        # Write inside torch.no_grad()
        bank.write(h, torch.ones(1), torch.ones(1))

        # Stored slot tensors must not have grad_fn attached
        assert bank.mem_keys.grad_fn is None, "mem_keys buffer should not have autograd history"
        assert bank.mem_vals.grad_fn is None, "mem_vals buffer should not have autograd history"
        assert bank.mem_importance.grad_fn is None, "mem_importance buffer should not have autograd history"
        assert bank.mem_confidence.grad_fn is None, "mem_confidence buffer should not have autograd history"
        assert bank.mem_state.grad_fn is None, "mem_state buffer should not have autograd history"
