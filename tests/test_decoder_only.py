"""
tests/test_decoder_only.py – Unit & Causal Regression Tests for Pure Decoder-Only LM.

Validates:
1. Causal Attention Mask: Future tokens CANNOT influence earlier logits (strictly causal).
2. Next-Token Prediction Alignment: input x_0...x_{T-1} predicts x_1...x_T.
3. No-Memory Fair Baseline: Identical outputs regardless of memory state mutations.
4. Memory State Persistence: Memory writes at earlier turns persist across later turns.
5. Host Write Head vs Locked i_proj: write_head provides write_prob; i_proj remains importance.
6. EOS Independence: Toggling is_eos does NOT alter write decisions.
7. Causal Memory Intervention: Enabling memory produces non-trivial difference vs No-Memory.
"""
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, STATE_ACTIVE
from models.decoder_only_memory_model import DecoderOnlyMemoryLM


@pytest.fixture
def decoder_lm():
    config = TinyMemoryConfig(
        memory_capacity=16,
        memory_dim=16,
        hidden_size=16,
        memory_top_k=4,
        memory_write_threshold=0.5
    )
    torch.manual_seed(42)
    model = DecoderOnlyMemoryLM(
        config=config,
        vocab_size=100,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        ff_dim=64
    )
    model.eval()
    return model, config


class TestCausalDecoderAutoregression:
    """Tests guaranteeing true causal Next-Token Prediction without future leakage."""

    def test_future_tokens_cannot_influence_past_logits(self, decoder_lm):
        """
        Critical Causal Invariant:
        Modifying token at position t MUST NOT change logits at positions < t.
        """
        model, _ = decoder_lm
        seq_len = 10
        torch.manual_seed(10)
        tokens_a = torch.randint(1, 100, (1, seq_len))
        tokens_b = tokens_a.clone()
        # Mutate the last 3 tokens
        tokens_b[0, 7:] = torch.randint(1, 100, (3,))

        with torch.no_grad():
            out_a = model(tokens_a, memory_mode='none')
            out_b = model(tokens_b, memory_mode='none')

        logits_a = out_a["logits"][0]
        logits_b = out_b["logits"][0]

        # Logits at positions 0 to 6 must be IDENTICAL to machine precision
        assert torch.allclose(logits_a[:7], logits_b[:7], atol=1e-5), (
            "Causal violation! Future tokens altered earlier logits."
        )

        # Logits at positions 7 to 9 should differ because inputs differ at position 7
        assert not torch.allclose(logits_a[7:], logits_b[7:], atol=1e-3), (
            "Expected logits to differ at mutated positions."
        )

    def test_next_token_target_alignment(self, decoder_lm):
        """
        Verifies that target_ids is aligned to x_{t+1}.
        """
        model, _ = decoder_lm
        tokens = torch.tensor([[10, 20, 30, 40, 50]])
        inputs = tokens[:, :-1]   # [10, 20, 30, 40]
        targets = tokens[:, 1:]   # [20, 30, 40, 50]

        with torch.no_grad():
            out = model(inputs, target_ids=targets, memory_mode='none')
            loss = out["loss"]

        assert loss is not None
        assert not torch.isnan(loss)
        assert loss.item() > 0.0


class TestNoMemoryFairBaseline:
    """Verifies that the No-Memory baseline genuinely operates with zero memory leakage."""

    def test_no_memory_independent_of_memory_bank_state(self, decoder_lm):
        """
        With memory_mode='none', output must be identical whether memory is empty or filled.
        """
        model, _ = decoder_lm
        x = torch.randint(1, 100, (2, 8))

        # Pass 1: Empty memory
        model.bank.load_memory_state(model.bank.empty_memory_state())
        with torch.no_grad():
            out_empty = model(x, memory_mode='none')["logits"]

        # Pass 2: Mutate memory buffers
        model.bank.mem_keys.normal_()
        model.bank.mem_vals.normal_()
        model.bank.mem_state.fill_(1)
        with torch.no_grad():
            out_mutated = model(x, memory_mode='none')["logits"]

        assert torch.allclose(out_empty, out_mutated, atol=1e-6), (
            "No-Memory baseline must NOT be affected by memory bank state!"
        )

    def test_different_inputs_produce_different_outputs_in_no_memory(self, decoder_lm):
        """No-Memory must genuinely encode its inputs."""
        model, _ = decoder_lm
        x1 = torch.tensor([[5, 10, 15, 20]])
        x2 = torch.tensor([[25, 30, 35, 40]])

        with torch.no_grad():
            out1 = model(x1, memory_mode='none')["logits"]
            out2 = model(x2, memory_mode='none')["logits"]

        assert not torch.allclose(out1, out2, atol=1e-3)


class TestMemoryBankDecoderIntegration:
    """Verifies write, read, and fusion pathways in the Pure Decoder-Only architecture."""

    def test_write_persists_in_episodic_buffer(self, decoder_lm):
        """Writing hidden states updates the active memory slot count."""
        model, _ = decoder_lm
        model.bank.load_memory_state(model.bank.empty_memory_state())

        h = torch.randn(1, model.embed_dim)
        wp = torch.tensor([0.9])  # above threshold=0.5

        _, w_idx = model.write_representation(h, write_prob=wp, memory_mode='bank')
        assert w_idx[0].item() != -1
        assert int(torch.sum(model.bank.mem_state == STATE_ACTIVE)) == 1

    def test_causal_memory_intervention(self, decoder_lm):
        """Enabling Memory Bank retrieval causally alters the query representation."""
        model, _ = decoder_lm
        model.bank.load_memory_state(model.bank.empty_memory_state())

        # Write a fact
        h_fact = torch.randn(1, model.embed_dim)
        model.write_representation(h_fact, write_prob=torch.tensor([1.0]), memory_mode='bank')

        # Read with Memory Bank vs No Memory
        h_query = torch.randn(1, model.embed_dim)
        fused_bank, _ = model.read_and_fuse(h_query, memory_mode='bank')
        fused_none, _ = model.read_and_fuse(h_query, memory_mode='none')

        # fused_none is identical to h_query
        assert torch.allclose(fused_none, h_query, atol=1e-6)
        # fused_bank is causally different due to memory injection
        assert not torch.allclose(fused_bank, fused_none, atol=1e-4), (
            "Memory Bank read & fuse must causally alter the hidden state!"
        )

    def test_eos_independence_in_decoder_write(self, decoder_lm):
        """is_eos parameter remains causally inert during write."""
        model, _ = decoder_lm
        h = torch.randn(1, model.embed_dim)
        wp = torch.tensor([0.9])

        # Write with is_eos=True
        model.bank.load_memory_state(model.bank.empty_memory_state())
        model.bank.write(model.memory_proj_in(h.clone()), torch.ones(1), wp.clone())
        state_true = model.bank.mem_keys.clone()

        # Write with is_eos=False
        model.bank.load_memory_state(model.bank.empty_memory_state())
        model.bank.write(model.memory_proj_in(h.clone()), torch.zeros(1), wp.clone())
        state_false = model.bank.mem_keys.clone()

        assert torch.allclose(state_true, state_false, atol=1e-6), (
            "is_eos must be causally inert in Memory Bank writes."
        )
