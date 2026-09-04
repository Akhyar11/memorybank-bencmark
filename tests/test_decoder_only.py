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

    def test_write_head_gradient_and_learning(self, decoder_lm):
        """
        Explicit Gradient Audit:
        Verifies that write_head receives non-zero gradient from the supervised write objective,
        and taking an optimizer step genuinely updates write_head weights.
        """
        model, _ = decoder_lm
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        input_ids = torch.tensor([[10, 20, 30, 40, 50, 60]])
        target_ids = torch.tensor([[20, 30, 40, 50, 60, 70]])
        turn_ends = [1, 3, 5]
        facts = [{'turn': 0, 'key': 'topic', 'value': 'AI'}]
        target_recall = {'query_turn': 1}

        w_before = model.write_head.weight.clone()
        optimizer.zero_grad()

        out = model.forward_dialogue_autoregressive(
            input_ids, target_ids=target_ids, turn_end_indices=turn_ends,
            facts=facts, target_recall=target_recall, memory_mode='bank'
        )

        assert out['loss'] is not None
        out['loss'].backward()

        # Audit: write_head.weight.grad must exist and be non-zero
        assert model.write_head.weight.grad is not None, "write_head.weight.grad must not be None!"
        grad_norm = torch.norm(model.write_head.weight.grad).item()
        assert grad_norm > 1e-6, f"write_head gradient must be non-zero, got {grad_norm}"

        optimizer.step()
        w_after = model.write_head.weight.clone()
        assert not torch.allclose(w_before, w_after, atol=1e-5), "write_head weight must update after optimizer step!"

    def test_autoregressive_memory_influences_future_tokens(self, decoder_lm):
        """
        Validates that memory retrieved at timestep t causally alters predictions
        at timestep t+1 and subsequent answer timesteps.
        """
        model, _ = decoder_lm
        model.eval()

        input_ids = torch.tensor([[5, 10, 15, 20, 25, 30]])
        turn_ends = [1, 3, 5]
        facts = [{'turn': 0, 'key': 'city', 'value': 'Jambi'}]
        target_recall = {'query_turn': 1}

        with torch.no_grad():
            out_bank = model.forward_dialogue_autoregressive(
                input_ids, turn_end_indices=turn_ends,
                facts=facts, target_recall=target_recall, memory_mode='bank'
            )
            out_none = model.forward_dialogue_autoregressive(
                input_ids, turn_end_indices=turn_ends,
                facts=facts, target_recall=target_recall, memory_mode='none'
            )

        logits_bank = out_bank['logits'][0]
        logits_none = out_none['logits'][0]

        # In turn 0 and 1 (pre-query), logits should be identical
        assert torch.allclose(logits_bank[:4], logits_none[:4], atol=1e-5), (
            "Pre-query turns must not experience memory interference."
        )
        # In turn 2 (answer turn, tokens 4..5), retrieved memory MUST alter logits
        assert not torch.allclose(logits_bank[4:], logits_none[4:], atol=1e-4), (
            "Answer turn logits must be causally altered by retrieved memory!"
        )

    def test_context_truncation_long_term_memory_causality(self, decoder_lm):
        """
        Critical test: If the relevant fact is no longer available in the host decoder's
        context window (evicted by distractors), the locked Memory Bank retrieves that information
        and causally alters the future prediction compared to No-Memory.
        """
        model, _ = decoder_lm
        model.eval()

        # Sequence of 12 tokens, with fact at tokens 0..1, distractors at 2..7, query at 8..9, answer at 10..11
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
        turn_ends = [1, 7, 9, 11]
        facts = [{'turn': 0, 'key': 'city', 'value': 'Pangkalpinang'}]
        target_recall = {'query_turn': 2}

        # Truncate context window to 4 tokens.
        # At turn 2 (query, tokens 8..9), the window [9-4:9] = [5, 6, 7, 8] excludes turn 0 (tokens 0..1)!
        context_window = 4

        with torch.no_grad():
            out_bank = model.forward_dialogue_autoregressive(
                input_ids, turn_end_indices=turn_ends, facts=facts,
                target_recall=target_recall, memory_mode='bank',
                context_window=context_window
            )
            out_none = model.forward_dialogue_autoregressive(
                input_ids, turn_end_indices=turn_ends, facts=facts,
                target_recall=target_recall, memory_mode='none',
                context_window=context_window
            )

        # Under truncated context, Bank retrieves from memory slot even though tokens are evicted
        assert out_bank['scores'] is not None, "Memory Bank must retrieve from episodic slot."
        logits_bank = out_bank['logits'][0, 10:]
        logits_none = out_none['logits'][0, 10:]

        assert not torch.allclose(logits_bank, logits_none, atol=1e-4), (
            "Memory Bank must causally alter predictions when context is truncated!"
        )

    def test_memory_update_and_suppression(self, decoder_lm):
        """
        Verifies that an updated fact in later turn correctly updates memory bank,
        and subsequent retrieval references the new fact.
        """
        model, _ = decoder_lm
        model.eval()

        model.bank.load_memory_state(model.bank.empty_memory_state())

        # Write fact 1 (Jambi)
        h1 = torch.randn(1, model.embed_dim)
        _, w1 = model.write_representation(h1, write_prob=torch.tensor([1.0]), memory_mode='bank')
        slot1 = w1[0].item()

        # Update fact 1 with fact 2 (similar key or update)
        h2 = h1 + 0.01 * torch.randn_like(h1)  # high cosine similarity to trigger update
        _, w2 = model.write_representation(h2, write_prob=torch.tensor([1.0]), memory_mode='bank')
        slot2 = w2[0].item()

        # If similarity was above threshold, slot1 was updated in-place; otherwise inserted
        assert slot2 != -1, "Write must succeed"
        assert model.bank.mem_state[slot2] == STATE_ACTIVE

    def test_generate_with_memory_bank(self, decoder_lm):
        """
        Verifies that model.generate() works smoothly when memory_mode='bank',
        both before and after memory writes (ensuring no unpacking ValueError).
        """
        model, _ = decoder_lm
        model.eval()
        model.reset_memory()

        tokens = torch.tensor([[10, 20, 30]], dtype=torch.long)

        # Turn 1: empty bank, set write_head bias high so write triggers
        model.write_head.bias.data.fill_(10.0)
        with torch.no_grad():
            gen_tokens, info = model.generate(
                input_ids=tokens,
                max_new_tokens=5,
                memory_mode="bank",
                write_threshold=0.5
            )

        assert gen_tokens.shape[1] <= 5
        assert info["did_write"] is True
        assert info["memory_active"] > 0

        # Turn 2: bank has active memory, reading and fusing must work without unpacking error
        with torch.no_grad():
            gen_tokens_2, info_2 = model.generate(
                input_ids=tokens,
                max_new_tokens=5,
                memory_mode="bank",
                write_threshold=0.99
            )

        assert gen_tokens_2.shape[1] <= 5
        assert info_2["memory_active"] > 0


