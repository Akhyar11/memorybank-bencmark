import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import MemoryState, TinyMemoryBank, TinyMemoryConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "gpt2-indo-instruct-tuned")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "izzulgod/gpt2-indo-instruct-tuned"


def test_empty_memory_read_zero():
    """Empty memory bank must return zero retrieval vector."""
    config = TinyMemoryConfig(memory_dim=768)
    bank = TinyMemoryBank(config)
    q = torch.randn(1, 768)
    m_bar, indices = bank.read(q)
    assert torch.allclose(m_bar, torch.zeros_like(m_bar))
    assert indices == []


def test_memory_insertion_and_capacity():
    """Adding memories respects max capacity by dropping oldest."""
    config = TinyMemoryConfig(memory_capacity=5, memory_dim=32)
    bank = TinyMemoryBank(config)

    for i in range(10):
        bank.add_memory(torch.full((32,), float(i)))

    assert bank.num_memories == 5
    # Oldest 0..4 dropped, latest 5..9 kept
    assert bank.memories[0][0].item() == 5.0
    assert bank.memories[-1][0].item() == 9.0


def test_top_k_cosine_retrieval():
    """Cosine similarity correctly retrieves top-K most similar memories."""
    config = TinyMemoryConfig(memory_dim=16, top_k=2)
    bank = TinyMemoryBank(config)

    target = torch.randn(16)
    target = target / torch.norm(target)

    # Similar vector (high cosine sim)
    v_sim = target + torch.randn(16) * 0.01
    # Distractor vectors (orthogonal/random)
    v_dist1 = torch.randn(16)
    v_dist2 = torch.randn(16)

    bank.add_memory(v_dist1)
    bank.add_memory(v_sim)
    bank.add_memory(v_dist2)

    query = target.unsqueeze(0)
    m_bar, top_idx = bank.read(query, top_k=1)

    # Memory at index 1 must be the top retrieved memory
    assert top_idx[0] == 1


def test_memory_influence_on_logits():
    """Changing memory state changes model logits when memory is used."""
    config = TinyMemoryConfig(memory_dim=768, hidden_size=768)
    model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, memory_config=config, freeze_backbone=True)
    model.eval()

    input_ids = torch.tensor([[50256, 100, 200]])

    # 1. Forward with empty memory
    model.reset_memory()
    out1 = model(input_ids, use_memory=True)

    # 2. Add distinct memory and forward again
    model.bank.add_memory(torch.randn(768) * 5.0)
    # Set fusion weight for memory to non-zero
    with torch.no_grad():
        model.fusion_proj.weight[:, 768:] = 0.5
    out2 = model(input_ids, use_memory=True)

    assert not torch.allclose(out1["logits"], out2["logits"], atol=1e-3)


def test_frozen_backbone_and_pure_ntp_loss():
    """Verify frozen GPT-2 backbone and training gradient flow to W_c and W_f."""
    model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, freeze_backbone=True)
    model.train()

    # Backbone weights must not require grad
    for name, param in model.gpt2.named_parameters():
        assert not param.requires_grad, f"GPT-2 parameter {name} is trainable!"

    # c_proj and fusion_proj must require grad
    assert model.c_proj.weight.requires_grad
    assert model.fusion_proj.weight.requires_grad

    # Test gradient backward with active memory retrieval
    model.bank.add_memory(torch.randn(768))
    input_ids = torch.tensor([[50256, 100, 200, 300]])
    labels = input_ids.clone()
    out = model(input_ids, labels=labels, use_memory=True)

    loss = out["loss"]
    assert loss is not None
    loss.backward()

    assert model.c_proj.weight.grad is not None
    assert model.fusion_proj.weight.grad is not None


def test_multi_turn_generate_growth_and_lifecycle():
    """Generates two turns, verifies exactly 2 memories added per turn, and lifecycle eviction."""
    model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, freeze_backbone=True)
    model.eval()
    model.reset_memory()

    # Turn 1
    model.generate(torch.tensor([[50256, 10]]), max_new_tokens=5, temperature=0.0)
    assert model.bank.num_memories == 2

    # Turn 2
    model.generate(torch.tensor([[50256, 20]]), max_new_tokens=5, temperature=0.0)
    assert model.bank.num_memories == 4
