"""
tests/test_matrix_memory.py
===========================
Comprehensive verification for GPT + Differentiable Memory Matrix:
  1. Memory Matrix M is non-trainable state (requires_grad = False).
  2. Pure analytical gradient verification: d(m)/d(q) == (1 / sqrt(d)) * M^T @ M.
  3. Continuous gradient flow from loss to W_q and W_f without vanishing gradient.
  4. Empty memory slot zero-contribution verification.
  5. FIFO replacement when 128 capacity is reached.
  6. End-to-end forward, backward, and generation.
"""

import math
import os
import sys
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.matrix_memory_bank import DifferentiableMemoryMatrix
from models.gpt2_matrix_memory_model import GPT2MatrixMemoryModel

MODEL_PATH = os.path.join(PROJECT_ROOT, "gpt2-indo-instruct-tuned")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "izzulgod/gpt2-indo-instruct-tuned"


class TestDifferentiableMemoryMatrix:
    """Unit tests for DifferentiableMemoryMatrix module."""

    def test_memory_properties_and_requires_grad(self):
        bank = DifferentiableMemoryMatrix(capacity=128, memory_dim=768)
        assert bank.M.shape == (128, 768)
        assert not bank.M.requires_grad, "M must NOT be trainable!"
        assert bank.num_memories == 0

    def test_empty_memory_returns_zero(self):
        bank = DifferentiableMemoryMatrix(capacity=128, memory_dim=768)
        q = torch.randn(2, 5, 768, requires_grad=True)
        m, s = bank.read(q)

        assert m.shape == (2, 5, 768)
        assert s.shape == (2, 5, 128)
        assert torch.allclose(m, torch.zeros_like(m))
        assert torch.allclose(s, torch.zeros_like(s))

    def test_analytical_gradient_flow_exactness(self):
        """
        Verifies that PyTorch autograd computes exactly:
            d(m) / d(q) = (1 / sqrt(d)) * M^T @ M
        without any distortion, clipping, or vanishing.
        """
        d = 64
        bank = DifferentiableMemoryMatrix(capacity=16, memory_dim=d, scaling=True)

        # Write 5 random memories
        for _ in range(5):
            bank.write(torch.randn(d))

        M = bank.M  # (16, d)
        q = torch.randn(1, d, requires_grad=True)

        m, _ = bank.read(q)  # (1, d)

        # Let L = v . m for a random vector v
        v = torch.randn(1, d)
        L = torch.sum(v * m)
        L.backward()

        # Analytical gradient of L with respect to q:
        # m = (1/sqrt(d)) * q @ M^T @ M
        # dL/dq = (1/sqrt(d)) * v @ (M^T @ M)^T = (1/sqrt(d)) * v @ M^T @ M
        scale = 1.0 / math.sqrt(d)
        expected_grad = scale * torch.matmul(torch.matmul(v, M.t()), M)

        assert q.grad is not None
        assert torch.allclose(q.grad, expected_grad, atol=1e-5), "Gradient does not match analytical M^T @ M projection!"

    def test_fifo_capacity_and_write(self):
        capacity = 8
        dim = 16
        bank = DifferentiableMemoryMatrix(capacity=capacity, memory_dim=dim)

        # Write 8 vectors
        for i in range(capacity):
            bank.write(torch.full((dim,), float(i)))

        assert bank.num_memories == 8
        assert bank.M[0, 0].item() == 0.0
        assert bank.M[-1, 0].item() == 7.0

        # Write 9th vector: slot 0 should be evicted (value 1.0 becomes first)
        bank.write(torch.full((dim,), 99.0))
        assert bank.num_memories == 8
        assert bank.M[0, 0].item() == 1.0
        assert bank.M[-1, 0].item() == 99.0


class TestGPT2MatrixMemoryModel:
    """Integration tests for GPT2MatrixMemoryModel."""

    @pytest.fixture
    def model_fixture(self):
        model = GPT2MatrixMemoryModel(
            model_name_or_path=MODEL_PATH,
            capacity=128,
            scaling=True,
            freeze_backbone=True,
        )
        return model

    def test_trainable_parameters(self, model_fixture):
        model = model_fixture
        # Backbone and M must be frozen
        assert not model.matrix_bank.M.requires_grad
        for name, p in model.gpt2.named_parameters():
            assert not p.requires_grad, f"GPT-2 parameter {name} is not frozen!"

        # Query encoder and fusion proj must be trainable
        assert model.query_encoder.weight.requires_grad
        assert model.fusion_proj.weight.requires_grad
        assert model.fusion_proj.bias.requires_grad

        trainable, total = model.print_trainable_parameters()
        expected_trainable = (768 * 768) + (768 * 1536 + 768)
        assert trainable == expected_trainable

    def test_end_to_end_forward_and_gradient(self, model_fixture):
        model = model_fixture
        model.train()
        model.reset_memory()

        # Add 2 memories
        model.matrix_bank.write(torch.randn(768))
        model.matrix_bank.write(torch.randn(768))

        input_ids = torch.tensor([[50256, 101, 102, 103, 104]])
        labels = input_ids.clone()

        out = model(input_ids, labels=labels, use_memory=True, prompt_len=3)
        loss = out["loss"]
        assert loss is not None
        assert not torch.isnan(loss)

        loss.backward()

        # Verify gradient flow
        assert model.query_encoder.weight.grad is not None
        assert model.fusion_proj.weight.grad is not None
        assert model.query_encoder.weight.grad.norm().item() > 0.0
        assert model.fusion_proj.weight.grad.norm().item() > 0.0

    def test_generate_and_memory_write(self, model_fixture):
        model = model_fixture
        model.eval()
        model.reset_memory()

        assert model.matrix_bank.num_memories == 0

        # Turn 1
        input_ids = torch.tensor([[50256, 100, 200]])
        out1 = model.generate(input_ids, max_new_tokens=5, temperature=0.0)

        # After Turn 1 completes: exactly 2 memories written (prompt + AI response)
        assert model.matrix_bank.num_memories == 2

        # Turn 2
        input_ids2 = torch.tensor([[50256, 300, 400]])
        out2 = model.generate(input_ids2, max_new_tokens=5, temperature=0.0)

        # After Turn 2 completes: exactly 4 memories written
        assert model.matrix_bank.num_memories == 4

    def test_slot_expansion_without_retraining(self, model_fixture):
        """Verifies checkpoint trained with 128 slots loads seamlessly into 1024 slots."""
        model_128 = model_fixture
        sd = model_128.state_dict()

        # Instantiate model with 1024 slots
        model_1024 = GPT2MatrixMemoryModel(
            model_name_or_path=MODEL_PATH,
            capacity=1024,
            scaling=True,
            freeze_backbone=True,
        )

        # Seamless loading without size mismatch
        model_1024.load_state_dict(sd)

        # Verify writing 500 memories into expanded capacity
        for _ in range(500):
            model_1024.matrix_bank.write(torch.randn(768))

        assert model_1024.matrix_bank.num_memories == 500

        # Verify forward pass
        out = model_1024(torch.tensor([[50256, 100]]), use_memory=True)
        assert out["logits"].shape == (1, 2, 50257)
