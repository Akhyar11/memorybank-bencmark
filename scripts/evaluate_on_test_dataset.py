"""
scripts/evaluate_on_test_dataset.py
===================================
Evaluasi formal model GPT-2 + TinyMemoryBank pada dataset uji held-out (dataset/conversations_test.jsonl).

Mengevaluasi:
1. Baseline: No Memory (Zero-shot GPT-2 / Empty Memory Bank)
2. Proposed: N=8 Soft-Attention Memory Bank
3. Metrik:
   - Ground Truth Exact Match (EM %)
   - Token F1-Score (%)
   - Slot Utilization Efficiency
   - Generation Latency (ms/token)
   - Evaluasi per Kategori Topik (tech, travel, correction, dll.)
"""
import os
import sys
import time
import json
import argparse
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import TinyMemoryConfig


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gt_tokens)
    return (2 * prec * rec / (prec + rec)) * 100.0


def check_exact_match(prediction: str, ground_truth: str) -> float:
    if not ground_truth:
        return 0.0
    pred_clean = prediction.lower().strip()
    gt_clean = ground_truth.lower().strip()
    if gt_clean in pred_clean:
        return 100.0
    return 0.0


def load_test_samples(data_path, num_samples=None):
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except Exception:
                    pass
    if num_samples is not None and num_samples > 0:
        samples = samples[:num_samples]
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate Memory Bank on Test Dataset")
    parser.add_argument("--test_file", type=str, default="dataset/conversations_test.jsonl")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gpt2_causal_memory_best.pt")
    parser.add_argument("--num_samples", type=int, default=50, help="Jumlah sampel dialog untuk diuji (default: 50, -1 untuk semua 501)")
    parser.add_argument("--max_new_tokens", type=int, default=25)
    parser.add_argument("--output_json", type=str, default="results/test_dataset_evaluation.json")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device pengujian : {device}")

    model_dir = "gpt2-indo-instruct-tuned"
    print("Memuat tokenizer dan backbone model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mem_config = TinyMemoryConfig(
        memory_capacity=128,
        memory_dim=768,
        hidden_size=768,
    )

    model = GPT2MemoryModel(model_name_or_path=model_dir, memory_config=mem_config).to(device)

    if os.path.exists(args.checkpoint):
        st = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "model_state_dict" in st:
            model.load_state_dict(st["model_state_dict"], strict=False)
        elif "adapter_state_dict" in st:
            model.load_state_dict(st["adapter_state_dict"], strict=False)
        else:
            model.load_state_dict(st, strict=False)
        print(f"✓ Checkpoint loaded: {args.checkpoint}")
    else:
        print(f"⚠️ Checkpoint tidak ditemukan di {args.checkpoint}, menggunakan bobot inisialisasi.")

    model.eval()

    limit = None if args.num_samples <= 0 else args.num_samples
    samples = load_test_samples(args.test_file, limit)
    print(f"Total sampel pengujian: {len(samples)} dialog dari {args.test_file}\n")

    methods = [
        ("No Memory", "no_memory"),
        ("Differentiable Causal Memory", "causal_memory")
    ]

    metrics = {
        m_key: {
            "em": [],
            "f1": [],
            "slots": [],
            "latency": [],
            "by_topic": {}
        }
        for _, m_key in methods
    }

    pfx_ids = tokenizer("User: ", return_tensors="pt")["input_ids"].to(device)
    pfx_len = pfx_ids.shape[1]

    for item in tqdm(samples, desc="Mengevaluasi Test Set"):
        topic = item.get("topic", "general")
        facts = item.get("facts", [])
        turns = item.get("turns", [])
        target_recall = item.get("target_recall", {})

        recall_q = target_recall.get("question", "Siapa namaku?")
        recall_a = target_recall.get("answer", "")
        ground_truth = target_recall.get("ground_truth", "")

        # Ekstraksi fact turns
        fact_turns = []
        if facts:
            for f in facts:
                t_idx = f.get("turn", 0)
                if t_idx < len(turns):
                    u_t = turns[t_idx]["content"]
                    ai_t = turns[t_idx + 1]["content"] if t_idx + 1 < len(turns) else "Baik."
                    fact_turns.append((u_t, ai_t))
        if not fact_turns and len(turns) >= 2:
            fact_turns.append((turns[0]["content"], turns[1]["content"]))

        query_prompt = f"User: {recall_q}\nAI:"
        enc_query = tokenizer(query_prompt, return_tensors="pt").to(device)
        q_len = enc_query["input_ids"].shape[1]

        for m_name, m_key in methods:
            model.reset_memory()

            # Write memory
            if m_key == "causal_memory":
                with torch.no_grad():
                    for u_text, ai_text in fact_turns:
                        w_enc = tokenizer(f"User: {u_text}\nAI: {ai_text}\n", max_length=128, truncation=True, return_tensors="pt").to(device)
                        model(w_enc["input_ids"], use_memory=True, persist_memory=True)

            slots_used = float(model.bank.mem_occupancy.sum().item())

            # Timed Generation
            t0 = time.perf_counter()
            with torch.no_grad():
                out_gen = model.generate(
                    input_ids=enc_query["input_ids"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.1,
                    top_k=20,
                    stop_token_ids=[199]
                )
            t1 = time.perf_counter()

            gen_tokens = out_gen[0][q_len:]
            num_tokens = max(1, len(gen_tokens))
            latency_ms = ((t1 - t0) / num_tokens) * 1000.0

            prediction = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            em_score = check_exact_match(prediction, ground_truth)
            f1_score = compute_token_f1(prediction, recall_a)

            metrics[m_key]["em"].append(em_score)
            metrics[m_key]["f1"].append(f1_score)
            metrics[m_key]["slots"].append(slots_used)
            metrics[m_key]["latency"].append(latency_ms)

            if topic not in metrics[m_key]["by_topic"]:
                metrics[m_key]["by_topic"][topic] = {"em": [], "f1": []}
            metrics[m_key]["by_topic"][topic]["em"].append(em_score)
            metrics[m_key]["by_topic"][topic]["f1"].append(f1_score)

    print("\n" + "=" * 76)
    print(f"      HASIL EVALUASI FORMAL: HELD-OUT TEST DATASET ({len(samples)} DIALOG)")
    print("=" * 76)
    print(f"{'Metode / Arsitektur':<28} | {'Exact Match':<12} | {'Token F1':<10} | {'Avg Slots':<10} | {'Latency':<10}")
    print("-" * 76)

    summary = {}
    for m_name, m_key in methods:
        mean_em = float(sum(metrics[m_key]["em"]) / len(metrics[m_key]["em"]))
        mean_f1 = float(sum(metrics[m_key]["f1"]) / len(metrics[m_key]["f1"]))
        mean_slots = float(sum(metrics[m_key]["slots"]) / len(metrics[m_key]["slots"]))
        mean_lat = float(sum(metrics[m_key]["latency"]) / len(metrics[m_key]["latency"]))

        summary[m_key] = {
            "name": m_name,
            "mean_em": mean_em,
            "mean_f1": mean_f1,
            "mean_slots": mean_slots,
            "mean_latency_ms": mean_lat
        }
        print(f"{m_name:<28} | {mean_em:>10.2f}% | {mean_f1:>8.2f}% | {mean_slots:>10.1f} | {mean_lat:>7.1f} ms")
    print("=" * 76)

    # Rincian Topik
    print("\n--- Rincian Kinerja per Topik Dialog (Token F1 / Exact Match) ---")
    all_topics = sorted(list(metrics["no_memory"]["by_topic"].keys()))
    print(f"{'Topik / Kategori':<32} | {'No Memory (F1/EM)':<20} | {'Proposed N=8 (F1/EM)':<20}")
    print("-" * 76)
    for top in all_topics:
        no_top_f1 = sum(metrics["no_memory"]["by_topic"][top]["f1"]) / max(1, len(metrics["no_memory"]["by_topic"][top]["f1"]))
        no_top_em = sum(metrics["no_memory"]["by_topic"][top]["em"]) / max(1, len(metrics["no_memory"]["by_topic"][top]["em"]))
        p_top_f1 = sum(metrics["causal_memory"]["by_topic"][top]["f1"]) / max(1, len(metrics["causal_memory"]["by_topic"][top]["f1"]))
        p_top_em = sum(metrics["causal_memory"]["by_topic"][top]["em"]) / max(1, len(metrics["causal_memory"]["by_topic"][top]["em"]))
        print(f"{top:<32} | {no_top_f1:5.1f}% / {no_top_em:4.1f}%     | {p_top_f1:5.1f}% / {p_top_em:4.1f}%")
    print("=" * 76)

    # Simpan ke JSON
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({
            "num_samples": len(samples),
            "summary": summary,
            "by_topic": {
                top: {
                    "no_memory": {
                        "f1": sum(metrics["no_memory"]["by_topic"][top]["f1"]) / max(1, len(metrics["no_memory"]["by_topic"][top]["f1"])),
                        "em": sum(metrics["no_memory"]["by_topic"][top]["em"]) / max(1, len(metrics["no_memory"]["by_topic"][top]["em"])),
                    },
                    "proposed_n8": {
                        "f1": sum(metrics["causal_memory"]["by_topic"][top]["f1"]) / max(1, len(metrics["causal_memory"]["by_topic"][top]["f1"])),
                        "em": sum(metrics["causal_memory"]["by_topic"][top]["em"]) / max(1, len(metrics["causal_memory"]["by_topic"][top]["em"])),
                    }
                }
                for top in all_topics
            }
        }, f, indent=2)

    print(f"\n✓ Hasil evaluasi disimpan ke: {args.output_json}")


if __name__ == "__main__":
    main()
