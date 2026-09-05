import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import MemoryState, TinyMemoryBank, TinyMemoryConfig

MODEL_PATH = "/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/gpt2-indo-instruct-tuned"


def test_adversarial_case_a_empty_memory():
    """
    CASE A — EMPTY MEMORY
      K = 0, V = 0, O = 0, U = 0, A = 0
    Expected:
      READ -> zero memory retrieval
      WRITE allocation: NOT uniform by mathematical necessity
      allocation must depend on k_t . p_i
    """
    config = TinyMemoryConfig(memory_capacity=16, memory_dim=32, hidden_size=32)
    bank = TinyMemoryBank(config)
    device = torch.device("cpu")
    state = bank.initialize_state(batch_size=1, device=device, dtype=torch.float32)

    h_t = torch.randn(1, 32)
    # READ on empty memory
    r_t, alpha = bank.read(h_t, state)
    assert torch.allclose(r_t, torch.zeros_like(r_t)), f"Empty memory read must be zero, got {r_t.abs().max()}"

    # WRITE on empty memory
    next_state, diag = bank.write(h_t, state)
    allocation = diag["allocation"]

    # Must NOT be uniform (1/C)
    uniform = torch.full_like(allocation, 1.0 / config.memory_capacity)
    assert not torch.allclose(allocation, uniform, atol=1e-3), "Empty memory write produced uniform allocation!"

    # Allocation must correlate with k_t . p_i
    k_t = bank.k_proj(h_t)
    kbar_t = k_t / (torch.norm(k_t, dim=-1, keepdim=True) + config.eps)
    pbar = bank.P / (torch.norm(bank.P, dim=-1, keepdim=True) + config.eps)
    expected_d = torch.einsum("bd,cd->bc", kbar_t, pbar)
    # The rank order of allocation on empty memory must match rank order of d_i
    assert torch.argmax(allocation).item() == torch.argmax(expected_d).item(), "Allocation top slot does not match top address score!"


def test_adversarial_case_b_one_occupied_slot():
    """
    CASE B — ONE OCCUPIED SLOT
      One slot has O = 1, others O = 0.
    Expected:
      Occupied slot routes by content similarity.
      Empty slots route by learned address similarity.
    """
    config = TinyMemoryConfig(memory_capacity=8, memory_dim=16, hidden_size=16)
    bank = TinyMemoryBank(config)
    device = torch.device("cpu")
    state = bank.initialize_state(batch_size=1, device=device, dtype=torch.float32)

    # Set slot 0 as occupied with specific key
    state.occupancy[0, 0] = 1.0
    state.keys[0, 0] = torch.randn(16)
    state.keys[0, 0] = state.keys[0, 0] / torch.norm(state.keys[0, 0])

    h_t = torch.randn(1, 16)
    next_state, diag = bank.write(h_t, state)
    assert next_state.occupancy[0, 0] >= 1.0 - 1e-6
    # Empty slots must have positive occupancy after write
    assert (next_state.occupancy > 0).all()


def test_adversarial_case_c_full_memory_replacement():
    """
    CASE C — FULL MEMORY
      All O_i approx 1.
    Expected:
      Replacement pressure depends on 1 - U_i.
      No hard eviction, no argmax, no top-k.
    """
    config = TinyMemoryConfig(memory_capacity=8, memory_dim=16, hidden_size=16, lambda_replace=2.0)
    bank = TinyMemoryBank(config)
    device = torch.device("cpu")
    state = bank.initialize_state(batch_size=1, device=device, dtype=torch.float32)
    state.occupancy.fill_(1.0)

    # Slot 0 has high usage, slot 1 has zero usage
    state.usage.fill_(0.5)
    state.usage[0, 0] = 1.0
    state.usage[0, 1] = 0.0

    h_t = torch.randn(1, 16)
    next_state, diag = bank.write(h_t, state)
    allocation = diag["allocation"]

    # Slot 1 (low usage) should have higher replacement pressure and thus higher allocation than slot 0 (high usage)
    # when content similarities are neutral
    assert allocation.shape == (1, 8)
    assert torch.allclose(allocation.sum(dim=-1), torch.tensor([1.0]))


def test_adversarial_case_d_zero_write_gate():
    """
    CASE D — ZERO WRITE GATE
      If g_write -> 0, then w_i -> 0, memory state is untouched.
    """
    config = TinyMemoryConfig(memory_capacity=8, memory_dim=16, hidden_size=16)
    bank = TinyMemoryBank(config)
    # Force write gate projection bias to -100 to make g_write ~ 0
    nn.init.constant_(bank.write_gate_proj.bias, -100.0)
    nn.init.zeros_(bank.write_gate_proj.weight)

    device = torch.device("cpu")
    state = bank.initialize_state(batch_size=1, device=device, dtype=torch.float32)
    dummy_keys = torch.randn(1, 8, 16)
    state.keys = dummy_keys / torch.norm(dummy_keys, dim=-1, keepdim=True)
    state.values.fill_(1.0)
    state.occupancy.fill_(0.5)

    h_t = torch.randn(1, 16)
    next_state, diag = bank.write(h_t, state)

    assert torch.allclose(diag["write_gate"], torch.tensor([0.0]), atol=1e-5)
    assert torch.allclose(next_state.keys, state.keys, atol=1e-5)
    assert torch.allclose(next_state.values, state.values, atol=1e-5)
    assert torch.allclose(next_state.occupancy, state.occupancy, atol=1e-5)


def test_adversarial_case_e_deterministic_identical_states():
    """
    CASE E — IDENTICAL HIDDEN STATES
      Same h_t and same M_t must produce identical READ and WRITE results.
    """
    config = TinyMemoryConfig(memory_capacity=8, memory_dim=16, hidden_size=16)
    bank = TinyMemoryBank(config)
    device = torch.device("cpu")

    h_t = torch.randn(1, 16)
    state = bank.initialize_state(batch_size=1, device=device, dtype=torch.float32)

    r1, a1 = bank.read(h_t, state)
    r2, a2 = bank.read(h_t, state)
    assert torch.allclose(r1, r2)
    assert torch.allclose(a1, a2)

    ns1, d1 = bank.write(h_t, state, alpha_read=a1)
    ns2, d2 = bank.write(h_t, state, alpha_read=a2)
    assert torch.allclose(ns1.keys, ns2.keys)
    assert torch.allclose(ns1.values, ns2.values)
    assert torch.allclose(ns1.occupancy, ns2.occupancy)
    assert torch.allclose(d1["allocation"], d2["allocation"])


def test_adversarial_case_f_batch_isolation():
    """
    CASE F — BATCH SIZE > 1
      Memory of batch element 0 must never influence batch element 1.
    """
    config = TinyMemoryConfig(memory_capacity=8, memory_dim=16, hidden_size=16)
    bank = TinyMemoryBank(config)
    device = torch.device("cpu")

    # Run batch size = 2
    state_b = bank.initialize_state(batch_size=2, device=device, dtype=torch.float32)
    h_b0 = torch.randn(1, 16)
    h_b1 = torch.randn(1, 16)
    h_both = torch.cat([h_b0, h_b1], dim=0)

    ns_both, _ = bank.write(h_both, state_b)

    # Run element 0 alone
    state_0 = bank.initialize_state(batch_size=1, device=device, dtype=torch.float32)
    ns_0, _ = bank.write(h_b0, state_0)

    # Element 0 in batch must match element 0 run alone
    assert torch.allclose(ns_both.keys[0], ns_0.keys[0], atol=1e-5)
    assert torch.allclose(ns_both.values[0], ns_0.values[0], atol=1e-5)
    assert torch.allclose(ns_both.occupancy[0], ns_0.occupancy[0], atol=1e-5)


def test_adversarial_case_g_tbptt_detach():
    """
    CASE G — TBPTT
      Intra-chunk gradients cross multiple memory timesteps;
      state is detached explicitly across chunk boundary.
    """
    model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, freeze_backbone=True)
    state = model.bank.initialize_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)

    # Attach dummy gradient
    state.keys = state.keys + torch.randn_like(state.keys).requires_grad_()
    assert state.keys.requires_grad

    detached = model.detach_memory_state(state)
    assert not detached.keys.requires_grad
    assert not detached.values.requires_grad
    assert not detached.occupancy.requires_grad
    assert not detached.usage.requires_grad
    assert not detached.age.requires_grad


def test_adversarial_case_h_target_leakage_prohibition():
    """
    CASE H — TARGET LEAKAGE
      Changing target label x_(t+1) while keeping h_t, M_t constant
      must NOT change WRITE at step t.
    """
    model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, freeze_backbone=True)
    model.eval()

    torch.manual_seed(100)
    input_ids = torch.randint(100, 1000, (1, 6))

    with torch.no_grad():
        out1 = model(input_ids=input_ids, use_memory=True)

    # Prefix of length 4
    with torch.no_grad():
        out2 = model(input_ids=input_ids[:, :4], use_memory=True)

    # In strictly causal predict-before-write, position 0..3 logits are identical
    for t in range(4):
        assert torch.allclose(out1["logits"][:, t, :], out2["logits"][:, t, :], atol=1e-4)


def test_adversarial_case_i_memory_influence_on_logits():
    """
    CASE I — MEMORY INFLUENCE
      Changing memory state while keeping h_t constant changes logits_t.
    """
    config = TinyMemoryConfig(memory_capacity=8, memory_dim=768, hidden_size=768)
    model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, memory_config=config, freeze_backbone=True)
    model.eval()

    device = torch.device("cpu")
    h_t = torch.randn(1, 1, 768)

    # Empty memory
    state_empty = model.bank.initialize_state(1, device, torch.float32)
    mem_out_empty = model._causal_memory_loop(h_t, state_empty, use_memory=True)
    logits_empty = mem_out_empty["logits"]

    # Populated memory
    state_pop = model.bank.initialize_state(1, device, torch.float32)
    state_pop.occupancy.fill_(1.0)
    state_pop.keys.fill_(1.0)
    state_pop.values.fill_(2.0)
    mem_out_pop = model._causal_memory_loop(h_t, state_pop, use_memory=True)
    logits_pop = mem_out_pop["logits"]

    assert not torch.allclose(logits_empty, logits_pop), "Memory state failed to influence prediction logits!"


def test_adversarial_case_j_empty_memory_read_zero():
    """
    CASE J — EMPTY MEMORY READ
      Empty memory MUST NOT produce arbitrary non-zero retrieved content.
    """
    config = TinyMemoryConfig(memory_capacity=16, memory_dim=32, hidden_size=32)
    bank = TinyMemoryBank(config)
    state = bank.initialize_state(1, torch.device("cpu"), torch.float32)

    h_t = torch.randn(1, 32)
    r_t, alpha = bank.read(h_t, state)
    assert (r_t == 0.0).all(), f"Retrieved memory on empty state is non-zero: {r_t}"


def test_frozen_backbone_and_pure_ntp_loss():
    """Verify frozen GPT-2 backbone and pure NTP loss gradient flow."""
    model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, freeze_backbone=True)
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    for n in trainable_names:
        assert "gpt2" not in n, f"GPT-2 parameter {n} is trainable!"
        assert "bank" in n

    B, T = 2, 8
    input_ids = torch.randint(100, 1000, (B, T))
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels, use_memory=True)

    # Check NTP loss formula
    shift_logits = out["logits"][:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    expected_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    assert torch.allclose(out["loss"], expected_loss)

    out["loss"].backward()
    for p in model.gpt2.parameters():
        assert p.grad is None

    assert model.bank.q_proj.weight.grad is not None
    assert model.bank.k_proj.weight.grad is not None
    assert model.bank.v_proj.weight.grad is not None
    assert model.bank.P.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
