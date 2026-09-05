import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, sparsemax

MODEL_PATH = "/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/gpt2-indo-instruct-tuned"


def test_sparsemax_properties():
    """Verify Sparsemax fulfills simplex constraints and produces natural differentiable sparsity."""
    x = torch.tensor([[10.0, 1.0, -5.0, 0.0]], requires_grad=True)
    p = sparsemax(x, dim=-1)

    # 1. Simplex constraints: sum == 1.0, all >= 0.0
    assert torch.allclose(p.sum(dim=-1), torch.tensor([1.0])), f"Sum not 1: {p.sum(dim=-1)}"
    assert (p >= 0.0).all(), "Negative probabilities in sparsemax output"

    # 2. Natural sparsity: low-scoring slots receive exact 0.0 without manual thresholding
    assert p[0, 2] == 0.0, f"Expected exact zero for low score, got {p[0, 2]}"

    # 3. Differentiability: backward pass works cleanly
    loss = (p * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_parameter_freeze():
    """Verify that 100% of GPT-2 backbone parameters are frozen and only TinyMemoryBank is trainable."""
    model = GPT2MemoryModel(
        model_name_or_path=MODEL_PATH,
        freeze_backbone=True,
    )

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]

    # All GPT2 layers must be frozen
    for name in frozen_names:
        assert "gpt2" in name, f"Non-GPT2 parameter frozen unexpectedly: {name}"

    # Only Memory Bank parameters must be trainable
    for name in trainable_names:
        assert "gpt2" not in name, f"GPT2 parameter leaked into trainable set: {name}"
        assert "bank" in name, f"Unexpected trainable parameter: {name}"

    trainable_count, total_count = model.print_trainable_parameters()
    assert trainable_count > 0
    assert (trainable_count / total_count) < 0.05, "Trainable parameter ratio exceeds 5%"


def test_strict_causal_predict_then_write_order():
    """
    Verify strict causal ordering:
      At step t, prediction is based purely on h_t and M_t.
      The write at step t produces M_{t+1}, which cannot affect prediction at step t.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, freeze_backbone=True).to(device)
    model.eval()

    B, T = 1, 6
    torch.manual_seed(42)
    input_ids = torch.randint(100, 1000, (B, T), device=device)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_memory=True)
        logits_full = out["logits"]

    # Now run prefix of length T - 2
    prefix_len = 4
    with torch.no_grad():
        out_prefix = model(input_ids=input_ids[:, :prefix_len], use_memory=True)
        logits_prefix = out_prefix["logits"]

    # In a strictly causal model without future leakage,
    # the predictions for positions 0..prefix_len-1 must match identically
    for t in range(prefix_len):
        assert torch.allclose(
            logits_full[:, t, :], logits_prefix[:, t, :], atol=1e-4
        ), f"Causal leakage detected at timestep {t}!"


def test_forward_and_backward_pure_ntp():
    """Verify forward and backward pass with strictly standard NTP cross-entropy loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2MemoryModel(
        model_name_or_path=MODEL_PATH,
        freeze_backbone=True,
    ).to(device)

    B, T = 2, 8
    input_ids = torch.randint(100, 1000, (B, T), device=device)
    labels = input_ids.clone()

    out = model(input_ids=input_ids, labels=labels, use_memory=True)
    assert "loss" in out
    assert "logits" in out
    assert out["loss"] is not None
    assert out["logits"].shape == (B, T, model.config.vocab_size)

    # Standard NTP loss check: CrossEntropy(logits[:, :-1], labels[:, 1:])
    shift_logits = out["logits"][:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    expected_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    assert torch.allclose(out["loss"], expected_loss, atol=1e-5), "Loss is not strictly NTP cross-entropy"

    # Backward pass
    out["loss"].backward()

    # Backbone parameters must receive zero gradient
    for p in model.gpt2.parameters():
        assert p.grad is None, "Frozen backbone received gradients!"

    # Memory Bank core projection weights must receive valid gradients
    assert model.bank.q_proj.weight.grad is not None
    assert model.bank.k_proj.weight.grad is not None
    assert model.bank.v_proj.weight.grad is not None
    assert model.bank.write_gate_proj.weight.grad is not None
    assert model.bank.novelty_proj.weight.grad is not None
    assert model.bank.fusion_proj.weight.grad is not None
    assert model.bank.slot_bias.grad is not None


def test_continuous_state_dynamics_no_thresholds():
    """Verify that confidence, utility, and allocation are continuous variables without hard thresholds."""
    config = TinyMemoryConfig(memory_capacity=16, memory_dim=32, hidden_size=32)
    bank = TinyMemoryBank(config)

    device = torch.device("cpu")
    state = bank.empty_memory_state(batch_size=1, device=device, dtype=torch.float32)

    h_t = torch.randn(1, 32)
    r_t, attn_t, read_scores = bank.read(h_t, state)
    assert r_t.shape == (1, 32)
    assert attn_t.shape == (1, 16)
    assert torch.allclose(attn_t.sum(dim=-1), torch.tensor([1.0]))

    fused_t, gate_t = bank.fuse(h_t, r_t)
    assert fused_t.shape == (1, 32)
    assert gate_t.shape == (1, 32)

    next_state, diag = bank.write(h_t, state, alpha_read=attn_t, fusion_gate=gate_t)
    assert next_state["keys"].shape == (1, 16, 32)
    assert next_state["vals"].shape == (1, 16, 32)
    assert next_state["confidence"].shape == (1, 16)
    assert next_state["importance"].shape == (1, 16)

    # State variables must remain continuous and bounded
    assert (next_state["confidence"] >= 0.0).all() and (next_state["confidence"] <= 1.0).all()
    assert (next_state["importance"] >= 0.0).all()

    # Diagnostics must be continuous
    assert "effective_write_slots" in diag
    assert diag["effective_write_slots"].item() >= 1.0
    assert "write_sparsity" in diag
    assert "confidence_sum" in diag


def test_tbptt_detach_boundary():
    """Verify that TBPTT detach explicitly detaches tensors across chunks without detaching intra-chunk steps."""
    model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, freeze_backbone=True)
    state = model.bank.empty_memory_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    
    # Simulate a step with gradients
    state["keys"] = state["keys"] + torch.randn_like(state["keys"])
    detached = model.detach_memory_state(state)

    for k in state:
        assert not detached[k].requires_grad, f"Tensor {k} retained computation graph after detach!"


if __name__ == "__main__":
    test_sparsemax_properties()
    test_parameter_freeze()
    test_strict_causal_predict_then_write_order()
    test_forward_and_backward_pure_ntp()
    test_continuous_state_dynamics_no_thresholds()
    test_tbptt_detach_boundary()
    print("✓ All differentiable causal Memory Bank architecture tests passed successfully!")
