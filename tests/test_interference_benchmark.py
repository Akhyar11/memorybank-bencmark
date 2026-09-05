"""
tests/test_interference_benchmark.py
======================================
Interference & Distractor Resistance Benchmark for Differentiable Causal Memory Bank & GPT-2.

Evaluates memory persistence against:
1. Low-Noise & High-Noise Distractors (verifying target memory remains robust).
2. Orthogonal Distractors (injection of other vectors).
3. Conversational Distractor Test.
"""
import os
import sys

sys.path.insert(0, "/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark")

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
    bank.q_proj.weight.data.copy_(bank.k_proj.weight.data)
    bank.reset_memory()

    torch.manual_seed(42)
    dim = cfg.memory_dim
    h_target = torch.randn(1, dim)
    h_target = h_target / torch.norm(h_target)

    # 1. Write target to memory state
    state = bank.empty_memory_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    state, write_diag = bank.write(h_target, state)
    target_alloc = write_diag["allocation"]
    top_target_slot = torch.argmax(target_alloc[0]).item()
    print(f"✓ Target Fact written with top slot #{top_target_slot}")

    # Read before distractors
    v_read_before, attn_before, scores_before = bank.read(h_target, state)
    expected_v = bank.v_proj(h_target)
    sim_before = F.cosine_similarity(v_read_before, expected_v).item()
    top_read_slot_before = torch.argmax(attn_before[0]).item()
    print(f"  └─ Similarity before distractors: {sim_before:.4f} (Top Slot #{top_read_slot_before})")

    # 2. Inject distractors
    test_cases = [
        ("Orthogonal Distractors (Distinct Topics)", 15, None),
        ("Low-Noise Distractors (Similar Information, noise=0.1)", 10, 0.1),
        ("Medium-Noise Distractors (Near Information, noise=0.3)", 15, 0.3),
        ("High-Noise Distractors (Noisy Information, noise=0.5)", 20, 0.5),
    ]

    for name, num_dist, noise in test_cases:
        bank.reset_memory()
        state = bank.empty_memory_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        state, _ = bank.write(h_target, state)

        for _ in range(num_dist):
            if noise is None:
                d_vec = torch.randn(1, dim)
            else:
                d_vec = h_target + torch.randn(1, dim) * noise
            d_vec = d_vec / torch.norm(d_vec)
            state, _ = bank.write(d_vec, state)

        v_read_after, attn_after, scores_after = bank.read(h_target, state)
        sim_after = F.cosine_similarity(v_read_after, expected_v).item()
        top_slot_after = torch.argmax(attn_after[0]).item()
        eff_slots = 1.0 / (attn_after.pow(2).sum(dim=-1).item() + 1e-8)

        status = "PASSED" if sim_after > 0.4 else "DEGRADED"
        print(f"\n[Test]: {name} ({num_dist} distractors)")
        print(f"  Effective read slots  : {eff_slots:.2f}/{cfg.memory_capacity}")
        print(f"  Target retrieval rank : Slot #{top_slot_after} (Initial: #{top_target_slot})")
        print(f"  Cosine Similarity     : {sim_after:.4f} (Before: {sim_before:.4f})")
        print(f"  Status                : {status}")

    print("\n✓ [LEVEL 1 BENCHMARK]: Vector interference test completed!\n")


def run_conversational_interference_benchmark():
    print("=" * 68)
    print("   [LEVEL 2]: CONVERSATIONAL INTERFERENCE (END-TO-END GPT-2)")
    print("=" * 68)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/gpt2-indo-instruct-tuned"
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

    print(f"✓ Target Fact: '{target_fact}' written into persistent causal memory.")
    print(f"  └─ Confidence Sum: {model.bank.mem_confidence.sum().item():.2f}")

    # 2. Inject distractors
    distractors = [
        "Hari ini cuaca di luar sangat panas dan terik.",
        "Aku baru saja memasak nasi goreng untuk sarapan pagi.",
        "Pesawat terbang menggunakan prinsip gaya angkat aerodinamika.",
        "Indonesia memiliki ribuan pulau dari Sabang sampai Merauke.",
    ]

    print(f"\n[Injecting {len(distractors)} conversational distractors...]")
    for i, d in enumerate(distractors):
        enc_d = tok(f"User: {d}\n", return_tensors="pt").to(device)
        with torch.no_grad():
            model(enc_d["input_ids"], use_memory=True, persist_memory=True)
        print(f"  Distractor #{i+1:02d}: '{d[:35]}...'")

    print(f"\nConfidence sum after distractors: {model.bank.mem_confidence.sum().item():.2f}/128")

    # 3. Recall Target Query
    recall_query = "Siapa namaku dan apa pekerjaanku?"
    enc_q = tok(f"User: {recall_query}\nAI:", return_tensors="pt").to(device)
    with torch.no_grad():
        q_len = enc_q["input_ids"].shape[1]
        out_gen = model.generate(
            input_ids=enc_q["input_ids"],
            max_new_tokens=25,
            temperature=0.3,
            stop_token_ids=[199],
        )
    ans = tok.decode(out_gen[0][q_len:], skip_special_tokens=True).strip()

    print("\n[RECALL EVALUATION]:")
    print("-" * 68)
    print(f"Recall Query : {recall_query}")
    print(f"Model Answer : {ans}")


if __name__ == "__main__":
    run_vector_interference_benchmark()
