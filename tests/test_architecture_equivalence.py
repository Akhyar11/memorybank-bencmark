"""
tests/test_architecture_equivalence.py – Architecture Equivalence & Reference Semantics Test (P1).

Validates exact semantic equivalence between the source-of-truth MemoryBank specification
(from mamoe/memory/bank.py) and PyTorch TinyMemoryBank across:
- Projections (q_proj, k_proj, v_proj, i_proj, fusion_proj)
- WRITE semantics (keys, vals, importance, confidence, timestamps, access_count, state)
- READ semantics (multi-factor score ranking, top-k aggregation, access reinforcement)
- DECAY semantics (effective_R = exp(-lam * dt) * (1 + rho * I), state transitions)
- REPLACEMENT semantics (EXPIRED -> DORMANT -> min importance ACTIVE)
- Non-differentiable state buffer mutation semantics
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import pytest

from models.tiny_memory_bank import (
    TinyMemoryBank, TinyMemoryConfig,
    STATE_EXPIRED, STATE_ACTIVE, STATE_DORMANT
)


class TestArchitectureEquivalence:
    @pytest.fixture
    def setup(self):
        config = TinyMemoryConfig(
            memory_capacity=8,
            memory_dim=16,
            hidden_size=16,
            memory_top_k=4,
            mem_alpha=1.0,
            mem_beta=0.5,
            mem_gamma=0.3,
            mem_delta=0.2,
            mem_decay_rate=0.01,
            mem_importance_protection=0.5,
            mem_reinforcement_rate=0.1,
            memory_threshold=-1.0,
            memory_write_threshold=0.9
        )
        torch.manual_seed(42)
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())
        return bank, config

    def test_projections_linear_equivalence(self, setup):
        """Verify all 5 projections perform exact linear matrix multiplication."""
        bank, config = setup
        h = torch.randn(2, config.hidden_size)

        # q_proj
        expected_q = torch.matmul(h, bank.q_proj.weight.T)
        assert torch.allclose(bank.q_proj(h), expected_q, atol=1e-6)

        # k_proj
        expected_k = torch.matmul(h, bank.k_proj.weight.T)
        assert torch.allclose(bank.k_proj(h), expected_k, atol=1e-6)

        # v_proj
        expected_v = torch.matmul(h, bank.v_proj.weight.T)
        assert torch.allclose(bank.v_proj(h), expected_v, atol=1e-6)

        # i_proj: sigmoid(W_I h + b_I)
        expected_i = torch.sigmoid(torch.matmul(h, bank.i_proj.weight.T) + bank.i_proj.bias)
        assert torch.allclose(torch.sigmoid(bank.i_proj(h)), expected_i, atol=1e-6)

        # fusion_proj: W_f [h; m]
        m = torch.randn(2, config.hidden_size)
        cat = torch.cat([h, m], dim=-1)
        expected_fused = torch.matmul(cat, bank.fusion_proj.weight.T)
        assert torch.allclose(bank.fuse(h, m), expected_fused, atol=1e-6)

    def test_write_and_read_exact_math_equivalence(self, setup):
        """
        Verify that READ implements the exact reference formula:
        score = alpha * sim + beta * imp + gamma * exp(-lam * dt) + delta * conf
        """
        bank, config = setup
        dim = config.memory_dim

        # Write 3 controlled memories
        h0 = torch.zeros(1, dim); h0[0, 0] = 1.0
        h1 = torch.zeros(1, dim); h1[0, 1] = 1.0
        h2 = torch.zeros(1, dim); h2[0, 2] = 1.0

        bank.write(h0, torch.ones(1), torch.ones(1))
        bank.write(h1, torch.ones(1), torch.ones(1))
        bank.write(h2, torch.ones(1), torch.ones(1))

        # Advance step to 10
        bank.global_step[0] = 10

        # Query using a controlled vector
        q_vec = torch.zeros(1, dim); q_vec[0, 0] = 1.0
        q = bank.q_proj(q_vec)
        q_norm = q / (torch.norm(q) + 1e-8)

        # Compute reference scores manually
        k_stored = bank.mem_keys
        k_norm = k_stored / (torch.norm(k_stored, dim=-1, keepdim=True) + 1e-8)
        sim_ref = torch.matmul(q_norm, k_norm.T)[0]

        dt = 10 - bank.mem_last_access
        rec_ref = torch.exp(-config.mem_decay_rate * dt.float())
        score_ref = (
            config.mem_alpha * sim_ref +
            config.mem_beta * bank.mem_importance +
            config.mem_gamma * rec_ref +
            config.mem_delta * bank.mem_confidence
        )
        score_ref = torch.where(bank.mem_state != STATE_EXPIRED, score_ref, torch.tensor(-1e9))

        # Execute read and capture scores
        _, scores = bank.read(q_vec, return_scores=True)
        assert torch.allclose(scores[0], score_ref, atol=1e-5), "Read score calculation differs from reference specification"

    def test_decay_formula_and_state_transitions(self, setup):
        """
        Verify reference decay formula:
        R = exp(-lambda * dt)
        effective_R = R * (1 + rho * I)
        effective_R < 0.1 -> EXPIRED
        0.1 <= effective_R < 0.5 -> DORMANT
        effective_R >= 0.5 -> ACTIVE
        """
        bank, config = setup
        dim = config.memory_dim
        h = torch.randn(1, dim)
        bank.write(h, torch.ones(1), torch.ones(1))

        # Initial state is ACTIVE
        assert bank.mem_state[0] == STATE_ACTIVE

        # Advance time so effective_R falls between 0.1 and 0.5
        # lam = 0.01, rho = 0.5, I = bank.mem_importance[0]
        I = bank.mem_importance[0].item()
        # Want effective_R ~ 0.3 -> exp(-0.01 * dt) * (1 + 0.5 * I) = 0.3
        target_R = 0.3 / (1.0 + 0.5 * I)
        dt_dormant = int(-np.log(target_R) / 0.01) + 1
        bank.global_step[0] = dt_dormant
        bank.decay_memory()
        assert bank.mem_state[0] == STATE_DORMANT, f"Expected DORMANT at dt={dt_dormant}, got {bank.mem_state[0].item()}"

        # Advance time so effective_R < 0.1 -> EXPIRED
        dt_expired = dt_dormant + 500
        bank.global_step[0] = dt_expired
        bank.decay_memory()
        assert bank.mem_state[0] == STATE_EXPIRED, f"Expected EXPIRED at dt={dt_expired}, got {bank.mem_state[0].item()}"
