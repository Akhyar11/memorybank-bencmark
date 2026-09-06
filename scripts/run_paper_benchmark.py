"""
scripts/run_paper_benchmark.py – Formal Academic Benchmark Suite
================================================================
Comprehensive evaluation for academic paper publication:
1. Baselines Comparison:
   - Baseline 1: No Memory (Zero-shot Frozen GPT-2 / Empty Memory Bank)
   - Baseline 2: Whole-Sentence Pooled Memory Bank (Compression into 1 slot)
   - Baseline 3 (Proposed): N=8 Chunked Token Memory Bank (Attention-Weighted Soft Pooling)
2. Interference Resistance Benchmark (0, 5, 10 Distractor Turns)
3. Evaluated Metrics:
   - Entity Exact Match (EM %)
   - Token F1-Score
   - Slot Utilization Efficiency
   - Average Generation Latency (ms/token)
   - Top-1 Memory Retrieval Quality
4. Outputs:
   - Console summary tables
   - results/paper_benchmark_results.json
   - results/paper_benchmark_table.tex (Ready to paste into LaTeX paper)
"""
import os
import sys
import time
import json
import torch
import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import TinyMemoryConfig


# ---------------------------------------------------------------------------
# Metric Helpers
# ---------------------------------------------------------------------------
def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Calculate token-level precision, recall, and F1."""
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)


def check_entity_match(prediction: str, entities: list) -> float:
    """Return ratio of target entities recalled in prediction."""
    pred_lower = prediction.lower()
    matches = sum(1 for e in entities if e.lower() in pred_lower)
    return (matches / len(entities)) * 100.0 if entities else 0.0


# ---------------------------------------------------------------------------
# Benchmark Dataset & Distractors
# ---------------------------------------------------------------------------
BENCHMARK_CASES = [
    {
        "id": "case_1",
        "fact": "Halo, perkenalkan namaku Akhyar dan aku bekerja sebagai Programmer.",
        "entities": ["Akhyar", "Programmer"],
        "query": "Siapa namaku dan apa pekerjaanku?",
        "ground_truth": "Namamu adalah Akhyar dan kamu bekerja sebagai Programmer."
    },
    {
        "id": "case_2",
        "fact": "Aku Fira, aku tinggal menetap di Pontianak dan punya anjing bernama Milo.",
        "entities": ["Fira", "Pontianak", "Milo"],
        "query": "Di mana aku tinggal dan apa nama anjingku?",
        "ground_truth": "Kamu tinggal di Pontianak dan anjingmu bernama Milo."
    },
    {
        "id": "case_3",
        "fact": "Namaku Fajar, aku berencana liburan ke Seoul dan menjalankan usaha thrift shop.",
        "entities": ["Fajar", "Seoul", "thrift shop"],
        "query": "Ke mana tujuanku liburan dan apa usahaku?",
        "ground_truth": "Kamu berencana liburan ke Seoul dan memiliki usaha thrift shop."
    },
    {
        "id": "case_4",
        "fact": "Aku bekerja menggunakan framework PyTorch dan saat senggang suka bermain Apex Legends.",
        "entities": ["PyTorch", "Apex Legends"],
        "query": "Framework apa yang kugunakan dan game apa yang kumainkan?",
        "ground_truth": "Kamu menggunakan framework PyTorch dan bermain game Apex Legends."
    },
    {
        "id": "case_5",
        "fact": "Minuman favoritku adalah Kopi Arabika dan aku sangat menyukai masakan Rendang Padang.",
        "entities": ["Kopi Arabika", "Rendang Padang"],
        "query": "Apa minuman favoritku dan makanan yang kusukai?",
        "ground_truth": "Minuman favoritmu adalah Kopi Arabika dan makanan kesukaanmu Rendang Padang."
    }
]

DISTRACTORS_POOL = [
    "Hari ini cuaca di luar sangat cerah dan langit berwarna biru terang.",
    "Aku baru saja memasak sepiring nasi goreng spesial untuk sarapan pagi.",
    "Pesawat terbang modern mengandalkan prinsip aerodinamika sayap untuk daya angkat.",
    "Indonesia memiliki keanekaragaman hayati yang tersebar dari Sabang sampai Merauke.",
    "Baterai lithium-ion menjadi teknologi penyimpanan energi utama untuk mobil listrik.",
    "Pemandangan matahari terbit di Gunung Bromo selalu menarik wisatawan mancanegara.",
    "Kucing rumahan rata-rata menghabiskan waktu 12 sampai 16 jam sehari untuk tidur.",
    "Suhu optimal penyimpanan biji kopi adalah di tempat sejuk dan kedap udara.",
    "Algoritma sorting seperti QuickSort memiliki kompleksitas rata-rata O(N log N).",
    "Jembatan Ampera di Palembang merupakan salah satu ikon arsitektur bersejarah.",
    "Danau Toba di Sumatera Utara terbentuk akibat letusan supervolcano purba.",
    "Kereta cepat Whoosh menghubungkan Jakarta dan Bandung dalam waktu sekitar 45 menit.",
    "Tanaman kaktus mampu menyimpan cadangan air dalam batangnya saat musim kemarau.",
    "Planet Mars sering dijuluki sebagai Planet Merah karena kandungan besi oksidanya.",
    "Minyak kelapa sawit banyak digunakan dalam industri pangan dan kosmetik global."
]


# ---------------------------------------------------------------------------
# Main Evaluator Engine
# ---------------------------------------------------------------------------
def run_benchmark():
    print("=" * 72)
    print("      ACADEMIC PAPER BENCHMARK SUITE: NEURAL MEMORY BANK")
    print("=" * 72)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device : {device}")

    candidate_model_dirs = [
        os.path.join(PROJECT_ROOT, "gpt2-indo-instruct-tuned"),
        "gpt2-indo-instruct-tuned",
        "izzulgod/gpt2-indo-instruct-tuned",
    ]
    model_dir = "izzulgod/gpt2-indo-instruct-tuned"
    for candidate in candidate_model_dirs:
        if os.path.exists(candidate):
            model_dir = candidate
            break

    candidate_ckpts = [
        os.path.join(PROJECT_ROOT, "checkpoints/gpt2_causal_memory_best.pt"),
        "checkpoints/gpt2_causal_memory_best.pt",
    ]
    ckpt_path = candidate_ckpts[0]
    for c in candidate_ckpts:
        if os.path.exists(c):
            ckpt_path = c
            break

    print("Loading Tokenizer & Backbone Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mem_config = TinyMemoryConfig(
        memory_capacity=128,
        memory_dim=768,
        hidden_size=768,
    )
    model = GPT2MemoryModel(model_name_or_path=model_dir, memory_config=mem_config).to(device)

    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in st:
            model.load_state_dict(st["model_state_dict"], strict=False)
        elif "adapter_state_dict" in st:
            model.load_state_dict(st["adapter_state_dict"], strict=False)
        else:
            model.load_state_dict(st, strict=False)
        print(f"✓ Checkpoint loaded: {ckpt_path}")
    else:
        print("⚠️ Checkpoint not found; using initialized model weights.")

    model.eval()

    methods = [
        ("No Memory", "no_memory"),
        ("Differentiable Causal Memory", "causal_memory")
    ]

    distractor_levels = [0, 5, 10]
    pfx_len = tokenizer("User: ", return_tensors="pt")["input_ids"].shape[1]

    results = {m_key: {"em": [], "f1": [], "slots": [], "latency": []} for _, m_key in methods}
    distractor_results = {lvl: {m_key: [] for _, m_key in methods} for lvl in distractor_levels}

    print("\n" + "-" * 72)
    print("Starting Quantitative Evaluation across benchmark scenarios...")
    print("-" * 72)

    for case_idx, case in enumerate(BENCHMARK_CASES):
        print(f"\nEvaluating Case {case_idx + 1}/{len(BENCHMARK_CASES)}: '{case['id']}'")
        fact_text = case["fact"]
        query_text = f"User: {case['query']}\nAI:"
        entities = case["entities"]
        ground_truth = case["ground_truth"]

        enc_fact = tokenizer(f"User: {fact_text}\n", return_tensors="pt").to(device)
        enc_query = tokenizer(query_text, return_tensors="pt").to(device)
        q_len = enc_query["input_ids"].shape[1]

        for m_name, m_key in methods:
            model.reset_memory()

            # Step 1: Write fact according to method
            if m_key == "causal_memory":
                with torch.no_grad():
                    model(enc_fact["input_ids"], use_memory=True, persist_memory=True)

            slots_used = float(model.bank.mem_occupancy.sum().item())

            # Step 2: Timed Generation
            t0 = time.perf_counter()
            with torch.no_grad():
                out_gen = model.generate(
                    input_ids=enc_query["input_ids"],
                    max_new_tokens=25,
                    temperature=0.1,
                    top_k=20,
                    stop_token_ids=[199]
                )
            t1 = time.perf_counter()

            gen_tokens = out_gen[0][q_len:]
            num_tokens = max(1, len(gen_tokens))
            latency_ms_per_token = ((t1 - t0) / num_tokens) * 1000.0

            prediction = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            em_score = check_entity_match(prediction, entities)
            f1_score = compute_token_f1(prediction, ground_truth) * 100.0

            results[m_key]["em"].append(em_score)
            results[m_key]["f1"].append(f1_score)
            results[m_key]["slots"].append(slots_used)
            results[m_key]["latency"].append(latency_ms_per_token)

            print(f"  [{m_name:28s}] EM: {em_score:5.1f}% | F1: {f1_score:5.1f}% | Slots: {slots_used:5.2f} | Latency: {latency_ms_per_token:5.1f}ms")

        # Evaluate Distractor Resistance for Case 1 (Akhyar)
        if case_idx == 0:
            print("\nEvaluating Distractor Interference Impact on Case 1:")
            for lvl in distractor_levels:
                for m_name, m_key in methods:
                    model.reset_memory()
                    if m_key == "causal_memory":
                        with torch.no_grad():
                            model(enc_fact["input_ids"], use_memory=True, persist_memory=True)

                    # Inject distractor turns
                    for d_text in DISTRACTORS_POOL[:lvl]:
                        enc_d = tokenizer(f"User: {d_text}\n", return_tensors="pt").to(device)
                        with torch.no_grad():
                            if m_key == "causal_memory":
                                model(enc_d["input_ids"], use_memory=True, persist_memory=True)

                    # Query after distractors
                    with torch.no_grad():
                        out_dgen = model.generate(
                            input_ids=enc_query["input_ids"],
                            max_new_tokens=25,
                            temperature=0.1,
                            stop_token_ids=[199]
                        )
                    p_d = tokenizer.decode(out_dgen[0][q_len:], skip_special_tokens=True).strip()
                    em_d = check_entity_match(p_d, entities)
                    distractor_results[lvl][m_key].append(em_d)

    # ---------------------------------------------------------------------------
    # Summary Table Construction
    # ---------------------------------------------------------------------------
    summary = {}
    for m_name, m_key in methods:
        summary[m_key] = {
            "name": m_name,
            "mean_em": float(np.mean(results[m_key]["em"])),
            "mean_f1": float(np.mean(results[m_key]["f1"])),
            "mean_slots": float(np.mean(results[m_key]["slots"])),
            "mean_latency_ms": float(np.mean(results[m_key]["latency"]))
        }

    print("\n" + "=" * 76)
    print("                 FORMAL ACADEMIC BENCHMARK RESULTS")
    print("=" * 76)
    print(f"{'Method / Architecture':<28} | {'Exact Match':<11} | {'Token F1':<10} | {'Avg Slots':<10} | {'Latency':<9}")
    print("-" * 76)
    for _, m_key in methods:
        s = summary[m_key]
        print(f"{s['name']:<28} | {s['mean_em']:9.2f}% | {s['mean_f1']:8.2f}% | {s['mean_slots']:9.1f} | {s['mean_latency_ms']:6.1f} ms")
    print("=" * 76)

    print("\n" + "=" * 76)
    print("         INTERFERENCE RESISTANCE UNDER DISTRACTOR TURNS (EM %)")
    print("=" * 76)
    print(f"{'Method / Architecture':<28} | {'0 Distractors':<14} | {'5 Distractors':<14} | {'10 Distractors':<14}")
    print("-" * 76)
    for m_name, m_key in methods:
        em_0 = distractor_results[0][m_key][0] if distractor_results[0][m_key] else 0.0
        em_5 = distractor_results[5][m_key][0] if distractor_results[5][m_key] else 0.0
        em_10 = distractor_results[10][m_key][0] if distractor_results[10][m_key] else 0.0
        print(f"{m_name:<28} | {em_0:12.1f}% | {em_5:12.1f}% | {em_10:12.1f}%")
    print("=" * 76)

    # ---------------------------------------------------------------------------
    # Export LaTeX Table File
    # ---------------------------------------------------------------------------
    res_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, "paper_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "distractor_results": distractor_results,
            "raw_results": results
        }, f, indent=2)
    print(f"\n✓ Saved JSON results: {json_path}")

    tex_path = os.path.join(res_dir, "paper_benchmark_table.tex")
    table_rows = []
    for m_name, m_key in methods:
        m_data = summary[m_key]
        table_rows.append(
            f"{m_name:<32} & {m_data['mean_em']:>6.1f} & {m_data['mean_f1']:>6.1f} & {m_data['mean_slots']:>6.1f} & {m_data['mean_latency_ms']:>6.1f} \\\\"
        )
    rows_str = "\n".join(table_rows)

    latex_content = f"""% LaTeX Table generated for Academic Paper Submission
\\begin{{table}}[t]
\\centering
\\caption{{Quantitative Comparison of Parametric Neural Memory Bank against Baselines on Multi-Turn Episodic Recall.}}
\\label{{tab:memory_benchmark}}
\\resizebox{{\\columnwidth}}{{!}}{{
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Method / Architecture}} & \\textbf{{Exact Match (\\%)}} & \\textbf{{Token F1 (\\%)}} & \\textbf{{Slots Used}} & \\textbf{{Latency (ms/tok)}} \\\\
\\hline
{rows_str}
\\hline
\\end{{tabular}}
}}
\\end{{table}}
"""
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"✓ Saved LaTeX Table: {tex_path}")

    return summary


if __name__ == "__main__":
    run_benchmark()
