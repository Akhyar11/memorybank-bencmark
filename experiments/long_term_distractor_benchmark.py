"""
experiments/long_term_distractor_benchmark.py – Long-Term Memory & Distractor Delay Benchmark.

Evaluates the central scientific question:
"If the relevant information is no longer available in the host decoder's effective context,
can the locked Memory Bank retrieve that information and causally increase the probability
of the correct future token?"

Controlled delays:
- delay = 0 distractor turns (immediate recall, within context)
- delay = 8 distractor turns (evicted from context window)
- delay = 32 distractor turns (deep long-term distractor)
- delay = 128 distractor turns (ultra long-term distractor)

Under a fixed context budget (e.g. W = 64 tokens), distractor turns push the initial fact
outside the host decoder's self-attention window.

Compares:
1. No Memory (Context-truncated causal LM baseline)
2. NN Memory (Nearest-Neighbor baseline)
3. Memory Bank (Locked Episodic Memory Bank)
"""
import os
import sys
import json
import collections
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tiny_memory_bank import TinyMemoryConfig
from models.decoder_only_memory_model import DecoderOnlyMemoryLM
from dataset.conversation_dataset import get_or_create_tokenizer
from evaluation.conversational_evaluator import ConversationalEvaluator, compute_string_em, compute_string_f1


DISTRACTOR_PAIRS = [
    ("Apa makanan kesukaanmu?", "Aku sangat suka makan nasi goreng dan sate ayam."),
    ("Cuaca di kotamu hari ini bagaimana?", "Hari ini cuaca cerah dan sedikit berangin."),
    ("Apakah kamu punya hobi tertentu?", "Aku senang mendengarkan musik dan membaca buku teknologi."),
    ("Bagaimana dengan proyek kerjamu?", "Proyek berjalan lancar sesuai target sprint mingguan."),
    ("Apakah kamu suka olahraga?", "Aku suka lari pagi setiap akhir pekan bersama teman."),
    ("Rekomendasi film apa yang bagus?", "Film sci-fi petualangan luar angkasa sangat seru ditonton."),
    ("Apakah kamu sering bepergian?", "Kadang-kadang saat libur panjang aku suka berwisata alam."),
    ("Apa bahasa pemrograman favoritmu?", "Python dan Rust adalah bahasa yang paling sering kugunakan.")
]


def generate_distractor_dialogue(
    entity_id: str,
    target_key: str,
    ground_truth: str,
    question: str,
    answer: str,
    num_distractors: int = 8
) -> Dict[str, Any]:
    """
    Generates a controlled dialogue with:
    - Turn 0: User introduces fact
    - Turn 1: Assistant acknowledges
    - Turns 2 .. 2*D+1: D distractor user/assistant turn pairs
    - Turn 2*D+2: User queries the fact
    - Turn 2*D+3: Assistant provides target answer
    """
    turns = [
        {"role": "user", "content": f"Halo, saya ingin memberitahu bahwa {target_key} saya adalah {ground_truth}."},
        {"role": "assistant", "content": f"Baik, saya catat bahwa {target_key} Anda adalah {ground_truth}."}
    ]

    for i in range(num_distractors):
        q, a = DISTRACTOR_PAIRS[i % len(DISTRACTOR_PAIRS)]
        turns.append({"role": "user", "content": q})
        turns.append({"role": "assistant", "content": a})

    query_turn = len(turns)
    turns.append({"role": "user", "content": question})
    turns.append({"role": "assistant", "content": answer})

    chatml = ""
    for t in turns:
        chatml += f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>\n"

    return {
        "id": f"distractor_d{num_distractors}_{entity_id}",
        "entity_id": entity_id,
        "topic": f"distractor_delay_{num_distractors}",
        "turns": turns,
        "chatml": chatml,
        "facts": [
            {
                "fact_id": f"fact_{target_key}",
                "turn": 0,
                "key": target_key,
                "value": ground_truth,
                "status": "active"
            }
        ],
        "target_recall": {
            "query_turn": query_turn,
            "target_key": target_key,
            "ground_truth": ground_truth,
            "question": question,
            "answer": answer
        }
    }


def run_long_term_benchmark(
    checkpoint_path: Optional[str] = None,
    tokenizer_path: str = "dataset/tokenizer.json",
    delays: Tuple[int, ...] = (0, 8, 32, 128),
    context_window: int = 64,
    embed_dim: int = 32,
    num_samples_per_delay: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 72)
    print("     LONG-TERM MEMORY & CONTEXT TRUNCATION DISTRACTOR BENCHMARK    ")
    print("=" * 72)
    print(f"DEVICE          : {device}")
    print(f"DELAYS (TURNS)  : {list(delays)}")
    print(f"CONTEXT WINDOW  : {context_window} tokens (Truncation Budget)")
    print(f"SAMPLES / DELAY : {num_samples_per_delay}")
    print("=" * 72)

    tok = get_or_create_tokenizer(tokenizer_path)
    vocab_size = tok.get_vocab_size()

    cfg = TinyMemoryConfig(
        memory_capacity=128, memory_dim=embed_dim, hidden_size=embed_dim,
        memory_write_threshold=0.5, mem_alpha=2.0
    )
    model = DecoderOnlyMemoryLM(
        config=cfg, vocab_size=vocab_size,
        embed_dim=embed_dim, num_layers=1, num_heads=2, ff_dim=216
    ).to(device)

    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded weights from {checkpoint_path}")

    model.eval()
    evaluator = ConversationalEvaluator(model, tok, device=device)

    test_cases = [
        ("domisili", "Pangkalpinang", "Di kota mana saya tinggal sekarang?", "Anda tinggal di kota Pangkalpinang."),
        ("pekerjaan", "Backend Engineer", "Apa profesi saya saat ini?", "Profesi Anda adalah Backend Engineer."),
        ("makanan_favorit", "Pempek Palembang", "Apa makanan kesukaan saya?", "Makanan kesukaan Anda adalah Pempek Palembang."),
        ("hobi", "Fotografi", "Apa kegiatan hobi saya?", "Hobi Anda adalah Fotografi."),
        ("universitas", "Institut Teknologi Bandung", "Di mana kampus tempat saya kuliah?", "Anda kuliah di Institut Teknologi Bandung.")
    ]

    modes = ["No Memory", "NN Memory", "Memory Bank"]
    mode_keys = {"No Memory": "none", "NN Memory": "nn", "Memory Bank": "bank"}

    results_by_delay = {}

    for delay in delays:
        print(f"\n--- Testing Distractor Delay = {delay} Turns (Context Window: {context_window} tokens) ---")
        items = []
        for i in range(num_samples_per_delay):
            case = test_cases[i % len(test_cases)]
            item = generate_distractor_dialogue(
                entity_id=f"user_{delay}_{i}",
                target_key=case[0],
                ground_truth=case[1],
                question=case[2],
                answer=case[3],
                num_distractors=delay
            )
            items.append(item)

        delay_summary = {}
        for m_name in modes:
            m_key = mode_keys[m_name]
            res = evaluator.evaluate_dataset(
                items, memory_mode=m_key,
                write_threshold=0.5, context_window=context_window
            )
            delay_summary[m_name] = res
            print(f"  [{m_name:12s}] R@1: {res['r1']:5.1f}% | R@5: {res['r5']:5.1f}% | MRR: {res['mrr']:.4f} | "
                  f"EM: {res['em']:4.1f}% | F1: {res['f1']:4.1f}% | Causal Action: {res['causal_intervention_rate']:5.1f}% | "
                  f"Prob Inc: {res.get('target_prob_increased_rate', 0.0):5.1f}%")

        results_by_delay[delay] = delay_summary

    print("\n" + "=" * 72)
    print("                    DISTRACTOR BENCHMARK SUMMARY TABLE                  ")
    print("=" * 72)
    print(f"{'Delay (Turns)':<14} | {'No Mem R@1':<10} | {'NN Mem R@1':<10} | {'Bank R@1':<10} | {'Bank Prob Inc':<14}")
    print("-" * 72)
    for delay in delays:
        r_none = results_by_delay[delay]["No Memory"]["r1"]
        r_nn = results_by_delay[delay]["NN Memory"]["r1"]
        r_bank = results_by_delay[delay]["Memory Bank"]["r1"]
        p_inc = results_by_delay[delay]["Memory Bank"].get("target_prob_increased_rate", 0.0)
        print(f"{delay:<14} | {r_none:<10.1f}% | {r_nn:<10.1f}% | {r_bank:<10.1f}% | {p_inc:<14.1f}%")
    print("=" * 72)

    return results_by_delay


if __name__ == "__main__":
    run_long_term_benchmark(delays=(0, 8, 32, 128), context_window=64)
