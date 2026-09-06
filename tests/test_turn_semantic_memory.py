import os
import sys
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import MemoryState, TinyMemoryBank, TinyMemoryConfig

MODEL_PATH = os.path.join(PROJECT_ROOT, "gpt2-indo-instruct-tuned")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "izzulgod/gpt2-indo-instruct-tuned"


def test_tiny_memory_bank_basic_read_write():
    cfg = TinyMemoryConfig(memory_dim=768, top_k=2)
    bank = TinyMemoryBank(cfg)

    # 1. Empty read returns zeros
    q = torch.randn(1, 768)
    m_bar, indices = bank.read(q)
    assert m_bar.shape == (1, 768)
    assert torch.allclose(m_bar, torch.zeros_like(m_bar))
    assert indices == []

    # 2. Add memories
    v1 = torch.randn(768)
    v2 = torch.randn(768)
    bank.add_memory(v1)
    bank.add_memory(v2)
    assert bank.num_memories == 2

    # 3. Read with Top-2
    m_bar, indices = bank.read(q, top_k=2)
    assert m_bar.shape == (1, 768)
    assert len(indices) == 2
    # Verify read count tracking
    assert sum(bank.read_counts) == 2


def test_lifecycle_eviction():
    cfg = TinyMemoryConfig(memory_dim=768, eviction_threshold_ratio=0.05, min_age_for_eviction=3)
    bank = TinyMemoryBank(cfg)

    # Add 4 memories
    for _ in range(4):
        bank.add_memory(torch.randn(768))

    # Age them all to 3 turns
    for _ in range(3):
        bank.step_turn()

    # Set artificial read counts:
    # Mean will be (100 + 100 + 100 + 1) / 4 = 75.25
    # Cutoff = 0.05 * 75.25 = 3.76
    # Item 4 has read count 1 < 3.76 -> must be evicted!
    bank.read_counts = [100, 100, 100, 1]

    evicted = bank.evict_lifecycle(threshold_ratio=0.05, min_age=3)
    assert evicted == 1
    assert bank.num_memories == 3
    assert bank.read_counts == [100, 100, 100]


def test_gpt2_memory_model_trainable_parameters():
    cfg = TinyMemoryConfig(memory_dim=768, hidden_size=768)
    model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, memory_config=cfg, freeze_backbone=True)

    trainable_params, total_params = model.print_trainable_parameters()
    # W_c is 768 * 768 = 589,824
    # W_f is 1536 * 768 = 1,179,648
    # Total trainable = 1,769,472
    assert trainable_params == (768 * 768) + (1536 * 768)
    assert trainable_params == 1769472

    # Ensure backbone is frozen
    assert not model.gpt2.transformer.wte.weight.requires_grad
    assert not model.gpt2.lm_head.weight.requires_grad
    assert model.c_proj.weight.requires_grad
    assert model.fusion_proj.weight.requires_grad


def test_gpt2_memory_model_generate_and_memory_growth():
    cfg = TinyMemoryConfig(memory_dim=768, hidden_size=768)
    model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, memory_config=cfg, freeze_backbone=True)
    model.eval()
    model.reset_memory()

    assert model.bank.num_memories == 0

    # Turn 1: generate
    input_ids = torch.tensor([[50256, 100, 200]])  # 3 tokens
    out = model.generate(input_ids, max_new_tokens=5, temperature=0.0)

    # Exactly 2 memories should be added in 1 turn!
    # Memory 1: User prompt last hidden state
    # Memory 2: AI response final token hidden state
    assert model.bank.num_memories == 2

    # Turn 2: another generate
    input_ids_2 = torch.tensor([[50256, 300, 400]])
    out2 = model.generate(input_ids_2, max_new_tokens=5, temperature=0.0)

    # Now there must be exactly 4 memories!
    assert model.bank.num_memories == 4


def test_read_count_matches_forward_passes():
    """
    Verifies that bank.read is called on EVERY forward pass:
    If there is 1 prompt forward + N decoding token forwards = 1 + N forwards,
    the read count must increase by exactly 1 + N.
    """
    cfg = TinyMemoryConfig(memory_dim=768, hidden_size=768, top_k=1)
    model = GPT2MemoryModel(model_name_or_path=MODEL_PATH, memory_config=cfg, freeze_backbone=True)
    model.eval()
    model.reset_memory()

    # Preload 1 memory so reading increments read_counts
    model.bank.add_memory(torch.randn(768))
    initial_reads = sum(model.bank.read_counts)

    # Generate tokens
    max_new_tokens = 5
    out = model.generate(torch.tensor([[50256, 100]]), max_new_tokens=max_new_tokens, temperature=0.0)

    # Number of generated tokens = len(out[0]) - input_len
    num_generated = out.shape[1] - 2
    # Total forward passes = 1 (prompt) + num_generated (tokens)
    expected_reads = 1 + num_generated
    actual_reads = sum(model.bank.read_counts) - initial_reads

    assert actual_reads == expected_reads, f"Expected {expected_reads} reads for {expected_reads} forwards, got {actual_reads}!"

