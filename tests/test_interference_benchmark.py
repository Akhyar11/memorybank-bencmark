"""
tests/test_interference_benchmark.py
======================================
Interference & Distractor Resistance Benchmark for Turn-Level Semantic Memory Bank & GPT-2.

Evaluates memory persistence against:
1. Low-Noise & High-Noise Distractors (verifying target memory remains robust).
2. Orthogonal Distractors (injection of other vectors).
3. Conversational Distractor Test.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig


def run_vector_interference_benchmark():
    print("=" * 68)
    print("   [LEVEL 1]: VECTOR INTERFERENCE & DISTRACTOR BENCHMARK")
    print("=" * 68)

    cfg = TinyMemoryConfig(memory_capacity=128, memory_dim=64, hidden_size=64)
    bank = TinyMemoryBank(cfg)
    bank.reset_memory()

    torch.manual_seed(42)
    dim = cfg.memory_dim
    h_target = torch.randn(dim)
    h_target = h_target / torch.norm(h_target)

    # 1. Write target to memory bank
    bank.add_memory(h_target)
    print(f"✓ Target Fact written to memory bank. Current memories: {bank.num_memories}")

    # Read before distractors
    q = h_target.unsqueeze(0)
    v_read_before, top_idx_before = bank.read(q, top_k=1)
    sim_before = F.cosine_similarity(v_read_before, q).item()
    print(f"  └─ Cosine Similarity before distractors: {sim_before:.4f} (Top Index #{top_idx_before[0]})")
    assert sim_before > 0.99

    # 2. Inject distractors
    test_cases = [
        ("Orthogonal Distractors (Distinct Topics)", 15, None),
        ("Low-Noise Distractors (Similar Information, noise=0.1)", 10, 0.1),
        ("Medium-Noise Distractors (Near Information, noise=0.3)", 15, 0.3),
        ("High-Noise Distractors (Noisy Information, noise=0.5)", 20, 0.5),
    ]

    for name, num_dist, noise in test_cases:
        bank.reset_memory()
        bank.add_memory(h_target)

        for _ in range(num_dist):
            if noise is None:
                d_vec = torch.randn(dim)
            else:
                d_vec = h_target + torch.randn(dim) * noise
            d_vec = d_vec / torch.norm(d_vec)
            bank.add_memory(d_vec)

        v_read_after, top_idx_after = bank.read(q, top_k=1)
        sim_after = F.cosine_similarity(v_read_after, q).item()
        target_retrieved = (top_idx_after[0] == 0)

        status = "PASSED" if (target_retrieved and sim_after > 0.8) else "DEGRADED"
        print(f"\n[Test]: {name} ({num_dist} distractors)")
        print(f"  Total memories in bank: {bank.num_memories}")
        print(f"  Target retrieval rank : #{top_idx_after[0]} (Initial: #0)")
        print(f"  Cosine Similarity     : {sim_after:.4f} (Before: {sim_before:.4f})")
        print(f"  Status                : {status}")
        assert target_retrieved, f"Target vector was not top retrieved in {name}!"

    print("\n✓ [LEVEL 1 BENCHMARK]: Vector interference test completed!\n")


def test_vector_interference():
    run_vector_interference_benchmark()


def test_conversational_interference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(PROJECT_ROOT, "gpt2-indo-instruct-tuned")
    if not os.path.exists(model_dir):
        model_dir = "izzulgod/gpt2-indo-instruct-tuned"
    tok = AutoTokenizer.from_pretrained(model_dir)

    mem_config = TinyMemoryConfig(
        memory_capacity=128,
        memory_dim=768,
        hidden_size=768,
    )
    model = GPT2MemoryModel(model_name_or_path=model_dir, memory_config=mem_config).to(device)
    model.eval()
    model.reset_memory()

    # 1. Target Fact
    target_fact = "Halo, perkenalkan namaku Akhyar dan aku bekerja sebagai Programmer."
    enc_target = tok(f"User: {target_fact}\n", return_tensors="pt").to(device)

    with torch.no_grad():
        out_t = model(enc_target["input_ids"], use_memory=True, persist_memory=True)

    assert model.bank.num_memories == 1

    # 2. Inject distractors
    distractors = [
        "Hari ini cuaca di luar sangat panas dan terik.",
        "Aku baru saja memasak nasi goreng untuk sarapan pagi.",
    ]
    for d in distractors:
        enc_d = tok(f"User: {d}\n", return_tensors="pt").to(device)
        with torch.no_grad():
            model(enc_d["input_ids"], use_memory=True, persist_memory=True)

    assert model.bank.num_memories == 1 + len(distractors)


if __name__ == "__main__":
    run_vector_interference_benchmark()
