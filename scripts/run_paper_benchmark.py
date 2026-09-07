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
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.gpt2_memory_model import GPT2MemoryModel
from models.gpt2_matrix_memory_model import GPT2MatrixMemoryModel
from models.tiny_memory_bank import TinyMemoryConfig
from models.seed import set_seed


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
# Benchmark Dataset Loader & Distractors Pool
# ---------------------------------------------------------------------------
def load_benchmark_cases(test_file: str, max_cases: int = 20):
    """
    Loads benchmark evaluation cases from JSONL (conversations_test.jsonl / val) or CSV (test.csv / val).
    Resolves relative paths against PROJECT_ROOT automatically.
    """
    resolved_path = test_file
    if not os.path.isabs(resolved_path) and not os.path.exists(resolved_path):
        candidates = [
            os.path.join(PROJECT_ROOT, test_file),
            os.path.join(PROJECT_ROOT, "dataset", os.path.basename(test_file)),
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"File benchmark test '{test_file}' tidak ditemukan di sistem!")

    cases = []
    if resolved_path.endswith(".jsonl"):
        with open(resolved_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                recall_meta = item.get("target_recall", {})
                if not recall_meta:
                    continue

                q_turn = recall_meta.get("query_turn", 8)
                turns = item.get("turns", [])

                history_turns = []
                for idx, t in enumerate(turns):
                    if idx >= q_turn:
                        break
                    role = t.get("role", "")
                    content = t.get("content", "").strip()
                    if content:
                        history_turns.append((role, content))

                question = recall_meta.get("question", "")
                target_entity = recall_meta.get("ground_truth", "")
                target_answer = recall_meta.get("answer", target_entity)

                if not question:
                    continue

                if target_entity == "UNKNOWN":
                    entities = ["belum pernah", "belum"]
                else:
                    entities = [target_entity] if isinstance(target_entity, str) else list(target_entity)

                cases.append({
                    "id": item.get("id", f"case_{len(cases)+1}"),
                    "topic": item.get("topic", "general"),
                    "history_turns": history_turns,
                    "query": question,
                    "entities": entities,
                    "ground_truth": target_answer,
                })
                if max_cases > 0 and len(cases) >= max_cases:
                    break

    elif resolved_path.endswith(".csv"):
        import csv
        with open(resolved_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fact = row.get("write_fact_A", "").strip()
                query = row.get("query_B", "").strip()
                ans = row.get("expected_output_A", "").strip()
                if fact and query and ans:
                    cases.append({
                        "id": row.get("fact_id", f"case_{len(cases)+1}"),
                        "topic": "fact_retrieval",
                        "history_turns": [("user", fact)],
                        "query": query,
                        "entities": [ans],
                        "ground_truth": f"Jawabannya adalah {ans}.",
                    })
                if max_cases > 0 and len(cases) >= max_cases:
                    break

    print(f"✓ Berhasil memuat {len(cases)} kasus evaluasi dari: {resolved_path}")
    return cases

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
def run_benchmark(
    test_file: str = "dataset/conversations_test.jsonl",
    max_cases: int = 20,
    model_dir_arg: str = None,
    output_dir: str = "results",
    verbose: bool = False,
    use_semantic_extractor: bool = False,
):
    print("=" * 72)
    print("      ACADEMIC PAPER BENCHMARK SUITE: NEURAL MEMORY BANK")
    print("=" * 72)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device : {device}")

    candidate_model_dirs = [
        model_dir_arg,
        os.path.join(PROJECT_ROOT, "gpt2-indo-instruct-tuned"),
        "gpt2-indo-instruct-tuned",
        "izzulgod/gpt2-indo-instruct-tuned",
    ]
    model_dir = "izzulgod/gpt2-indo-instruct-tuned"
    for candidate in candidate_model_dirs:
        if candidate and os.path.exists(candidate):
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
    model_causal = GPT2MemoryModel(model_name_or_path=model_dir, memory_config=mem_config).to(device)

    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in st:
            model_causal.load_state_dict(st["model_state_dict"], strict=False)
        elif "adapter_state_dict" in st:
            model_causal.load_state_dict(st["adapter_state_dict"], strict=False)
        else:
            model_causal.load_state_dict(st, strict=False)
        del st
        torch.cuda.empty_cache()
        print(f"✓ Causal Memory Checkpoint loaded: {ckpt_path}")
    else:
        print("⚠️ Causal Memory Checkpoint not found; using initialized model weights.")
    model_causal.eval()

    # Load Matrix Memory Model (Share frozen backbone to save 500MB VRAM!)
    model_matrix = GPT2MatrixMemoryModel(
        model_name_or_path=model_dir,
        capacity=128,
        scaling="dim",
        freeze_backbone=True,
    )
    model_matrix.gpt2 = model_causal.gpt2  # Shared backbone
    model_matrix.to(device)

    candidate_matrix_ckpts = [
        os.path.join(PROJECT_ROOT, "checkpoints", "matrix_adapter_best.pt"),
        os.path.join(PROJECT_ROOT, "checkpoints", "gpt2_semantic_matrix_best.pt"),
        os.path.join(PROJECT_ROOT, "checkpoints", "gpt2_matrix_memory_best.pt"),
    ]
    ckpt_matrix_path = None
    for cp in candidate_matrix_ckpts:
        if os.path.exists(cp):
            ckpt_matrix_path = cp
            break

    if ckpt_matrix_path:
        st_m = torch.load(ckpt_matrix_path, map_location="cpu", weights_only=False)
        if "adapter_state_dict" in st_m:
            model_matrix.load_adapter(st_m)
            print(f"✓ Matrix Memory Adapter (~7MB) loaded: {ckpt_matrix_path}")
        else:
            if "model_state_dict" in st_m:
                sd_m = dict(st_m["model_state_dict"])
            else:
                sd_m = dict(st_m)
            if "c_proj.weight" in sd_m and "query_encoder.weight" not in sd_m:
                sd_m["query_encoder.weight"] = sd_m["c_proj.weight"]
            model_matrix.load_state_dict(sd_m, strict=False)
            print(f"✓ Matrix Memory Checkpoint loaded: {ckpt_matrix_path}")
        del st_m
        torch.cuda.empty_cache()
    else:
        print("ℹ Matrix Memory Checkpoint not found; using initialized model weights.")

    if use_semantic_extractor:
        from models.semantic_extractor import SemanticSentenceExtractor
        print("Loading SemanticSentenceExtractor (IndoBERT)...")
        extractor = SemanticSentenceExtractor(extractor_type="indobert", device=device)
        model_matrix.set_semantic_extractor(extractor)
        print("✓ SemanticSentenceExtractor attached to Matrix Memory.")

    model_matrix.eval()

    methods = [
        ("No Memory", "no_memory"),
        ("Full In-Context GPT-2", "full_context"),
        ("Differentiable Causal Memory", "causal_memory"),
        ("Differentiable Matrix Memory", "matrix_memory"),
    ]

    distractor_levels = [0, 5, 10]
    results = {m_key: {"em": [], "f1": [], "slots": [], "latency": []} for _, m_key in methods}
    distractor_results = {lvl: {m_key: [] for _, m_key in methods} for lvl in distractor_levels}
    sample_predictions = []

    # Load evaluation cases dynamically from test/validation dataset
    cases = load_benchmark_cases(test_file=test_file, max_cases=max_cases)
    if not cases:
        raise ValueError(f"Tidak ada kasus pengujian yang berhasil dimuat dari {test_file}")

    print("\n" + "-" * 72)
    print(f"Starting Quantitative Evaluation across {len(cases)} test cases...")
    print(f"Dataset Source: {test_file}")
    print("-" * 72)

    pbar = tqdm(enumerate(cases), total=len(cases), desc="Benchmarking Cases", unit="case", dynamic_ncols=True)
    for case_idx, case in pbar:
        case_id = case["id"]
        history_turns = case.get("history_turns", [])
        query_text = f"User: {case['query']}\nAI:"
        entities = case["entities"]
        ground_truth = case["ground_truth"]

        enc_query = tokenizer(query_text, return_tensors="pt").to(device)
        q_len = enc_query["input_ids"].shape[1]

        if verbose:
            print(f"\n[Case {case_idx + 1}/{len(cases)}] ID: '{case_id}' (Topic: {case.get('topic', 'general')})")
            print(f"  Q : {case['query']}")
            print(f"  GT: {ground_truth} | Target Entities: {entities}")

        case_preds = {"id": case_id, "query": case["query"], "ground_truth": ground_truth, "entities": entities}

        for m_name, m_key in methods:
            if m_key == "no_memory":
                model_causal.reset_memory()
                slots_used = 0.0

                t0 = time.perf_counter()
                with torch.no_grad():
                    out_gen = model_causal.generate(
                        input_ids=enc_query["input_ids"],
                        max_new_tokens=30,
                        temperature=0.1,
                        top_k=20,
                        stop_token_ids=[199],
                        use_memory=False,
                    )
                t1 = time.perf_counter()
                gen_q_len = q_len

            elif m_key == "full_context":
                context_parts = []
                for role, content in history_turns:
                    pfx = "User: " if role.lower() == "user" else "AI: "
                    context_parts.append(f"{pfx}{content}")
                context_parts.append(query_text)
                full_prompt = "\n".join(context_parts)

                enc_full = tokenizer(full_prompt, return_tensors="pt").to(device)
                if enc_full["input_ids"].shape[1] > 1000:
                    enc_full["input_ids"] = enc_full["input_ids"][:, -1000:]
                    enc_full["attention_mask"] = enc_full["attention_mask"][:, -1000:]
                prompt_len_full = enc_full["input_ids"].shape[1]
                slots_used = 0.0

                t0 = time.perf_counter()
                with torch.no_grad():
                    out_gen = model_causal.gpt2.generate(
                        input_ids=enc_full["input_ids"],
                        attention_mask=enc_full["attention_mask"],
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                t1 = time.perf_counter()
                gen_q_len = prompt_len_full

            elif m_key == "causal_memory":
                model_causal.reset_memory()
                with torch.no_grad():
                    for role, content in history_turns:
                        pfx = "User: " if role.lower() == "user" else "AI: "
                        enc_t = tokenizer(f"{pfx}{content}\n", return_tensors="pt").to(device)
                        model_causal(enc_t["input_ids"], use_memory=True, persist_memory=True)
                slots_used = float(model_causal.bank.mem_occupancy.sum().item())

                t0 = time.perf_counter()
                with torch.no_grad():
                    out_gen = model_causal.generate(
                        input_ids=enc_query["input_ids"],
                        max_new_tokens=30,
                        temperature=0.1,
                        top_k=20,
                        stop_token_ids=[199],
                        use_memory=True,
                    )
                t1 = time.perf_counter()
                gen_q_len = q_len

            elif m_key == "matrix_memory":
                model_matrix.reset_memory()
                with torch.no_grad():
                    for role, content in history_turns:
                        pfx = "User: " if role.lower() == "user" else "AI: "
                        if model_matrix.semantic_extractor is not None:
                            model_matrix.write_semantic_text(f"{pfx}{content}")
                        else:
                            enc_t = tokenizer(f"{pfx}{content}\n", return_tensors="pt").to(device)
                            t_out = model_matrix.gpt2.transformer(enc_t["input_ids"], return_dict=True)
                            h_t = t_out.last_hidden_state[:, -1, :]
                            model_matrix.matrix_bank.write(h_t)
                slots_used = float(model_matrix.matrix_bank.num_memories)

                t0 = time.perf_counter()
                with torch.no_grad():
                    out_gen = model_matrix.generate(
                        input_ids=enc_query["input_ids"],
                        query_text=case["query"] if model_matrix.semantic_extractor is not None else None,
                        max_new_tokens=30,
                        temperature=0.1,
                        top_k=20,
                        stop_token_ids=[199],
                        use_memory=True,
                    )
                t1 = time.perf_counter()
                gen_q_len = q_len

            gen_tokens = out_gen[0][gen_q_len:]
            num_tokens = max(1, len(gen_tokens))
            latency_ms_per_token = ((t1 - t0) / num_tokens) * 1000.0

            prediction = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            if "\n" in prediction:
                prediction = prediction.split("\n")[0].strip()

            em_score = check_entity_match(prediction, entities)
            f1_score = compute_token_f1(prediction, ground_truth) * 100.0

            results[m_key]["em"].append(em_score)
            results[m_key]["f1"].append(f1_score)
            results[m_key]["slots"].append(slots_used)
            results[m_key]["latency"].append(latency_ms_per_token)
            case_preds[m_key] = {"prediction": prediction, "em": em_score, "f1": f1_score}

            if verbose:
                print(f"  [{m_name:28s}] EM: {em_score:5.1f}% | F1: {f1_score:5.1f}% | Slots: {slots_used:5.2f} | Latency: {latency_ms_per_token:5.1f}ms")
                print(f"     └─ Pred: \"{prediction}\"")

        sample_predictions.append(case_preds)

        pbar.set_postfix({
            "Causal_F1": f"{np.mean(results['causal_memory']['f1']):.1f}%",
            "FullCtx_F1": f"{np.mean(results['full_context']['f1']):.1f}%",
        })

        # Evaluate Distractor Resistance for Case 1
        if case_idx == 0:
            for lvl in distractor_levels:
                for m_name, m_key in methods:
                    if m_key == "no_memory":
                        model_causal.reset_memory()
                        with torch.no_grad():
                            out_dgen = model_causal.generate(
                                input_ids=enc_query["input_ids"],
                                max_new_tokens=30,
                                temperature=0.1,
                                top_k=20,
                                stop_token_ids=[199],
                                use_memory=False,
                            )
                        dgen_tokens = out_dgen[0][q_len:]

                    elif m_key == "full_context":
                        d_parts = []
                        for role, content in history_turns:
                            pfx = "User: " if role.lower() == "user" else "AI: "
                            d_parts.append(f"{pfx}{content}")
                        for d_text in DISTRACTORS_POOL[:lvl]:
                            d_parts.append(f"User: {d_text}")
                        d_parts.append(query_text)
                        full_d_prompt = "\n".join(d_parts)

                        enc_d = tokenizer(full_d_prompt, return_tensors="pt").to(device)
                        if enc_d["input_ids"].shape[1] > 1000:
                            enc_d["input_ids"] = enc_d["input_ids"][:, -1000:]
                            enc_d["attention_mask"] = enc_d["attention_mask"][:, -1000:]
                        d_prompt_len = enc_d["input_ids"].shape[1]

                        with torch.no_grad():
                            out_dgen = model_causal.gpt2.generate(
                                input_ids=enc_d["input_ids"],
                                attention_mask=enc_d["attention_mask"],
                                max_new_tokens=30,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )
                        dgen_tokens = out_dgen[0][d_prompt_len:]

                    elif m_key == "causal_memory":
                        model_causal.reset_memory()
                        with torch.no_grad():
                            for role, content in history_turns:
                                pfx = "User: " if role.lower() == "user" else "AI: "
                                enc_t = tokenizer(f"{pfx}{content}\n", return_tensors="pt").to(device)
                                model_causal(enc_t["input_ids"], use_memory=True, persist_memory=True)

                        # Inject distractor turns
                        for d_text in DISTRACTORS_POOL[:lvl]:
                            enc_d = tokenizer(f"User: {d_text}\n", return_tensors="pt").to(device)
                            with torch.no_grad():
                                model_causal(enc_d["input_ids"], use_memory=True, persist_memory=True)

                        # Query after distractors
                        with torch.no_grad():
                            out_dgen = model_causal.generate(
                                input_ids=enc_query["input_ids"],
                                max_new_tokens=30,
                                temperature=0.1,
                                top_k=20,
                                stop_token_ids=[199],
                                use_memory=True,
                            )
                        dgen_tokens = out_dgen[0][q_len:]

                    elif m_key == "matrix_memory":
                        model_matrix.reset_memory()
                        with torch.no_grad():
                            for role, content in history_turns:
                                pfx = "User: " if role.lower() == "user" else "AI: "
                                if model_matrix.semantic_extractor is not None:
                                    model_matrix.write_semantic_text(f"{pfx}{content}")
                                else:
                                    enc_t = tokenizer(f"{pfx}{content}\n", return_tensors="pt").to(device)
                                    t_out = model_matrix.gpt2.transformer(enc_t["input_ids"], return_dict=True)
                                    h_t = t_out.last_hidden_state[:, -1, :]
                                    model_matrix.matrix_bank.write(h_t)

                        # Inject distractor turns
                        for d_text in DISTRACTORS_POOL[:lvl]:
                            if model_matrix.semantic_extractor is not None:
                                model_matrix.write_semantic_text(f"User: {d_text}")
                            else:
                                enc_d = tokenizer(f"User: {d_text}\n", return_tensors="pt").to(device)
                                with torch.no_grad():
                                    d_out = model_matrix.gpt2.transformer(enc_d["input_ids"], return_dict=True)
                                    h_d = d_out.last_hidden_state[:, -1, :]
                                    model_matrix.matrix_bank.write(h_d)

                        # Query after distractors
                        with torch.no_grad():
                            out_dgen = model_matrix.generate(
                                input_ids=enc_query["input_ids"],
                                query_text=case["query"] if model_matrix.semantic_extractor is not None else None,
                                max_new_tokens=30,
                                temperature=0.1,
                                top_k=20,
                                stop_token_ids=[199],
                                use_memory=True,
                            )
                        dgen_tokens = out_dgen[0][q_len:]

                    p_d = tokenizer.decode(dgen_tokens, skip_special_tokens=True).strip()
                    if "\n" in p_d:
                        p_d = p_d.split("\n")[0].strip()
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

    print("\n" + "=" * 80)
    print("                 FORMAL ACADEMIC BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Method / Architecture':<30} | {'Exact Match':<11} | {'Token F1':<10} | {'Avg Slots':<10} | {'Latency':<9}")
    print("-" * 80)
    for _, m_key in methods:
        s = summary[m_key]
        print(f"{s['name']:<30} | {s['mean_em']:9.2f}% | {s['mean_f1']:8.2f}% | {s['mean_slots']:9.1f} | {s['mean_latency_ms']:6.1f} ms")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("         INTERFERENCE RESISTANCE UNDER DISTRACTOR TURNS (EM %)")
    print("=" * 80)
    print(f"{'Method / Architecture':<30} | {'0 Distractors':<14} | {'5 Distractors':<14} | {'10 Distractors':<14}")
    print("-" * 80)
    for m_name, m_key in methods:
        em_0 = distractor_results[0][m_key][0] if distractor_results[0][m_key] else 0.0
        em_5 = distractor_results[5][m_key][0] if distractor_results[5][m_key] else 0.0
        em_10 = distractor_results[10][m_key][0] if distractor_results[10][m_key] else 0.0
        print(f"{m_name:<30} | {em_0:12.1f}% | {em_5:12.1f}% | {em_10:12.1f}%")
    print("=" * 80)

    # ---------------------------------------------------------------------------
    # Export LaTeX Table File & JSON
    # ---------------------------------------------------------------------------
    res_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, "paper_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_file": test_file,
            "total_cases_evaluated": len(cases),
            "summary": summary,
            "distractor_results": distractor_results,
            "raw_results": results,
            "sample_predictions": sample_predictions,
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Academic Paper Benchmark Suite: Neural Memory Bank")
    parser.add_argument("--test_file", type=str, default="dataset/conversations_test.jsonl",
                        help="Path to test dataset (.jsonl or .csv)")
    parser.add_argument("--max_cases", type=int, default=10,
                        help="Max number of test cases to benchmark (default: 10, -1 for all)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path or name of pretrained model")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save output json and tex")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print verbose model predictions per case (default: False, progress bar only)")
    parser.add_argument("--use_semantic_extractor", action="store_true", default=False,
                        help="Enable IndoBERT semantic extractor for Matrix Memory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    set_seed(args.seed)

    run_benchmark(
        test_file=args.test_file,
        max_cases=args.max_cases,
        model_dir_arg=args.model_dir,
        output_dir=args.output_dir,
        verbose=args.verbose,
        use_semantic_extractor=args.use_semantic_extractor,
    )


if __name__ == "__main__":
    main()
