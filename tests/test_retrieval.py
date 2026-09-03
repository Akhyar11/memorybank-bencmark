"""
tests/test_retrieval.py – Retrieval metrics edge-cases & No-Memory baseline regression tests.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig
from models.transformer_qa_model import TransformerQAModel
from evaluation.metrics import recall_at_k, mean_reciprocal_rank, exact_match, batch_token_f1


class TestNoMemoryBaselineRegression:
    """PHASE 4: Regression tests ensuring No-Memory baseline has zero memory contribution."""

    def test_same_query_different_memory_produces_identical_output(self):
        """No-Memory must produce identical output regardless of what is stored in memory."""
        config = TinyMemoryConfig(memory_capacity=16, memory_dim=16, hidden_size=16)
        mdl = TransformerQAModel(config=config, vocab_size=50, embed_dim=16, num_layers=1, num_heads=2)
        mdl.eval()

        query_ids = torch.randint(1, 50, (2, 8))
        query_mask = torch.ones(2, 8)
        target_ids = torch.randint(1, 50, (2, 4))

        # Pass 1: Empty memory
        mdl.bank.load_memory_state(mdl.bank.empty_memory_state())
        out1, _, _, _ = mdl(query_ids, query_mask, torch.ones(2), torch.zeros(2), target_ids, memory_mode='none')

        # Pass 2: Fill memory with arbitrary data
        mdl.bank.mem_keys.normal_()
        mdl.bank.mem_vals.normal_()
        mdl.bank.mem_state.fill_(1)
        out2, _, _, _ = mdl(query_ids, query_mask, torch.ones(2), torch.zeros(2), target_ids, memory_mode='none')

        assert torch.allclose(out1, out2, atol=1e-5), "No-Memory output must NOT depend on memory state"

    def test_different_query_produces_different_output(self):
        """No-Memory must genuinely encode and respond to the query."""
        config = TinyMemoryConfig(memory_capacity=16, memory_dim=16, hidden_size=16)
        mdl = TransformerQAModel(config=config, vocab_size=50, embed_dim=16, num_layers=1, num_heads=2)
        mdl.eval()

        q1 = torch.tensor([[1, 2, 3, 4]])
        q2 = torch.tensor([[5, 6, 7, 8]])
        target_ids = torch.tensor([[9, 10]])

        out1, _, _, _ = mdl(q1, torch.ones_like(q1), torch.ones(1), torch.zeros(1), target_ids, memory_mode='none')
        out2, _, _, _ = mdl(q2, torch.ones_like(q2), torch.ones(1), torch.zeros(1), target_ids, memory_mode='none')

        assert not torch.allclose(out1, out2, atol=1e-3), "Different queries must produce different outputs in No-Memory"


class TestMetricsEdgeCases:
    """PHASE 14: Edge-case tests for recall_at_k and mean_reciprocal_rank."""

    def test_recall_at_k_exceeds_capacity(self):
        """K > memory capacity should gracefully evaluate without IndexError."""
        scores = np.array([0.1, 0.9, 0.4])  # capacity 3
        gt_idx = 1
        r = recall_at_k(scores, gt_idx, k_values=[1, 5, 10])
        assert r[1] == 1.0
        assert r[5] == 1.0
        assert r[10] == 1.0

    def test_missing_target_out_of_bounds(self):
        """gt_idx not present or invalid produces recall=0.0 and MRR=0.0."""
        scores = np.array([0.1, 0.9, 0.4])
        r = recall_at_k(scores, gt_idx=99, k_values=[1, 2])
        mrr = mean_reciprocal_rank(scores, gt_idx=99)
        assert r[1] == 0.0
        assert mrr == 0.0

    def test_empty_scores(self):
        """Empty scores array handled cleanly."""
        scores = np.array([])
        r = recall_at_k(scores, gt_idx=0, k_values=[1])
        mrr = mean_reciprocal_rank(scores, gt_idx=0)
        assert r[1] == 0.0
        assert mrr == 0.0

    def test_mrr_exact_ranking(self):
        """Verify exact reciprocal rank formula 1/rank."""
        scores = np.array([0.1, 0.8, 0.9, 0.3])
        # Ranked order: index 2 (0.9), index 1 (0.8), index 3 (0.3), index 0 (0.1)
        assert mean_reciprocal_rank(scores, gt_idx=2) == 1.0        # rank 1 -> 1/1 = 1.0
        assert mean_reciprocal_rank(scores, gt_idx=1) == 0.5        # rank 2 -> 1/2 = 0.5
        assert mean_reciprocal_rank(scores, gt_idx=3) == 1.0 / 3.0  # rank 3 -> 1/3
        assert mean_reciprocal_rank(scores, gt_idx=0) == 0.25       # rank 4 -> 1/4 = 0.25
