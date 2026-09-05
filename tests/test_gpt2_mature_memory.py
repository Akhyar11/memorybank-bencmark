import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from models.tiny_memory_bank import TinyMemoryConfig, STATE_ACTIVE
from models.gpt2_memory_model import GPT2MemoryModel

MODEL_PATH = "/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/gpt2-indo-instruct-tuned"

def test_parameter_freeze():
    """Memastikan 100% parameter GPT-2 dibekukan dan hanya TinyMemoryBank yang dilatih."""
    model = GPT2MemoryModel(
        model_name_or_path=MODEL_PATH,
        freeze_backbone=True
    )

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]

    # Pastikan semua layer GPT2 dibekukan
    for name in frozen_names:
        assert "gpt2" in name, f"Parameter non-GPT2 yang harusnya trainable malah beku: {name}"

    for name in trainable_names:
        assert "gpt2" not in name, f"Parameter GPT2 bocor ke trainable: {name}"
        assert any(k in name for k in ["bank", "write_head", "read_head", "memory_proj"]), f"Parameter tidak dikenal: {name}"

    trainable_count, total_count = model.print_trainable_parameters()
    assert trainable_count > 0
    # Parameter trainable harus di bawah 5%
    assert (trainable_count / total_count) < 0.05

def test_forward_and_backward():
    """Menguji forward pass dan kalkulasi gradien backward."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2MemoryModel(
        model_name_or_path=MODEL_PATH,
        freeze_backbone=True
    ).to(device)

    B, T = 2, 16
    input_ids = torch.randint(100, 1000, (B, T), device=device)
    labels = input_ids.clone()
    write_targets = torch.zeros(B, T, device=device)
    read_targets = torch.zeros(B, T, device=device)

    out = model(input_ids=input_ids, labels=labels, write_targets=write_targets, read_targets=read_targets)
    assert "loss" in out
    assert "logits" in out
    assert out["loss"] is not None
    assert out["logits"].shape == (B, T, model.config.vocab_size)

    # Backward pass
    out["loss"].backward()

    # Pastikan parameter GPT-2 sama sekali tidak menerima gradien
    for p in model.gpt2.parameters():
        assert p.grad is None

    # Pastikan modul TinyMemoryBank & Heads menerima gradien
    assert model.write_head.weight.grad is not None
    assert model.read_head.weight.grad is not None
    assert model.bank.fusion_proj.weight.grad is not None

def test_write_and_decay():
    """Menguji fungsi penulisan ke memory bank dan decay."""
    model = GPT2MemoryModel(
        model_name_or_path=MODEL_PATH,
        freeze_backbone=True
    )
    dim = model.memory_config.hidden_size
    h = torch.randn(dim)
    
    # Write ke slot
    model.write_memory(h, write_prob=torch.tensor([0.9]))
    active_slots = (model.bank.mem_state == STATE_ACTIVE).nonzero(as_tuple=True)[0]
    assert len(active_slots) == 1

    # Decay
    model.bank.global_step += 10
    model.bank.decay_memory()
    # State tidak boleh crash
    assert model.bank.mem_state[active_slots[0]] in [0, 1, 2]

if __name__ == "__main__":
    test_parameter_freeze()
    test_forward_and_backward()
    test_write_and_decay()
    print("✓ Seluruh pengujian GPT-2 + Mature TinyMemoryBank berhasil!")
