"""
scripts/benchmark_paper_metrics.py
==================================
Comprehensive Benchmark Pipeline for Paper/Report Metrics:
  1. Pure GPT-2 (Zero-shot / No Memory)
  2. Full-Context GPT-2 (Concatenated Conversation History)
  3. Proposed MemoryBank (W_q Adapter, scaling='dim')
  4. Proposed MemoryBank (W_q Adapter, scaling='sqrt' / Scaled Attention)

Evaluated Metrics:
  - Hit@1 Retrieval Accuracy (%)
  - Hit@3 Retrieval Accuracy (%)
  - MRR (Mean Reciprocal Rank)
  - Entity Recall (%) [Factual Slot Filling]
  - Entity Token-Overlap F1 (%)
  - IndoBERT BERTScore (Precision, Recall, F1 %)
  - IndoBERT Sentence Cosine Similarity
  - Avg. Input Tokens (Prompt Length)
  - Generation Latency (ms/turn)
"""

import os
import sys
import time
import json
import re
import string
import argparse
import math
from typing import List, Dict, Any, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.gpt2_matrix_memory_model import GPT2MatrixMemoryModel
from models.semantic_extractor import SemanticSentenceExtractor


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for fair entity and text comparison."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = " ".join(text.split())
    return text


def compute_entity_recall(prediction: str, ground_truth_entity: str) -> Tuple[float, float]:
    """
    Computes:
      1. Binary Entity Found (100.0 or 0.0): exact substring or >= 75% token recall.
      2. Token-level recall percentage (0.0 to 100.0).
    """
    clean_pred = normalize_text(prediction)
    clean_gt = normalize_text(ground_truth_entity)

    if not clean_gt:
        return 0.0, 0.0

    # Check exact substring
    if clean_gt in clean_pred:
        return 100.0, 100.0

    # Token-level overlap
    gt_tokens = set(clean_gt.split())
    pred_tokens = set(clean_pred.split())

    if not gt_tokens:
        return 0.0, 0.0

    common = gt_tokens & pred_tokens
    token_recall = (len(common) / len(gt_tokens)) * 100.0
    binary_found = 100.0 if token_recall >= 75.0 else 0.0

    return binary_found, token_recall


class BertScoreEvaluator:
    """Token-level contextual embedding alignment using shared IndoBERT (BERTScore formulation)."""

    def __init__(self, bert_model: Any, bert_tokenizer: Any, device: torch.device):
        self.device = device
        self.tokenizer = bert_tokenizer
        self.model = bert_model

    @torch.no_grad()
    def compute_bert_score(self, pred: str, ref: str) -> Tuple[float, float, float]:
        """Returns (Precision %, Recall %, F1 %)."""
        p_clean = pred.strip()
        r_clean = ref.strip()
        if not p_clean or not r_clean:
            return 0.0, 0.0, 0.0

        enc_p = self.tokenizer(p_clean, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        enc_r = self.tokenizer(r_clean, return_tensors="pt", truncation=True, max_length=128).to(self.device)

        out_p = self.model(**enc_p).last_hidden_state[0, 1:-1]  # drop [CLS], [SEP]
        out_r = self.model(**enc_r).last_hidden_state[0, 1:-1]

        if out_p.size(0) == 0 or out_r.size(0) == 0:
            return 0.0, 0.0, 0.0

        p_norm = F.normalize(out_p, p=2, dim=-1)
        r_norm = F.normalize(out_r, p=2, dim=-1)

        sim_matrix = torch.matmul(p_norm, r_norm.t())  # (P_len, R_len)

        r_score = sim_matrix.max(dim=0).values.mean().item()
        p_score = sim_matrix.max(dim=1).values.mean().item()
        f1 = (2 * p_score * r_score / (p_score + r_score + 1e-8)) * 100.0

        return p_score * 100.0, r_score * 100.0, f1

    @torch.no_grad()
    def compute_sentence_cosine(self, pred: str, ref: str) -> float:
        """Sentence-level cosine similarity of mean pooled representations."""
        if not pred.strip() or not ref.strip():
            return 0.0
        enc_p = self.tokenizer(pred, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        enc_r = self.tokenizer(ref, return_tensors="pt", truncation=True, max_length=128).to(self.device)

        h_p = self.model(**enc_p).last_hidden_state.mean(dim=1)
        h_r = self.model(**enc_r).last_hidden_state.mean(dim=1)

        cos = F.cosine_similarity(h_p, h_r, dim=-1).item()
        return max(0.0, cos) * 100.0


def load_dataset(file_path: str, max_samples: int = 50) -> List[Dict[str, Any]]:
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if "target_recall" in data and "turns" in data:
                        samples.append(data)
                except Exception:
                    pass
            if 0 < max_samples <= len(samples):
                break
    return samples


def run_benchmark_pipeline(
    test_file: str = "dataset/conversations_test.jsonl",
    checkpoint_path: str = "checkpoints/matrix_adapter_best.pt",
    model_name: str = "gpt2-indo-instruct-tuned",
    num_samples: int = 30,
    max_new_tokens: int = 30,
    device_str: str = "cuda",
    output_dir: str = "results",
) -> Dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print("=" * 85)
    print(" BENCHMARK MATRIKS PERBANDINGAN UNTUK PAPER / LAPORAN EVALUASI")
    print("=" * 85)
    print(f"Device Evaluasi     : {device}")
    print(f"Checkpoint Adapter  : {checkpoint_path}")
    print(f"Dataset Test        : {test_file}")
    print(f"Jumlah Sampel       : {num_samples}")
    print("=" * 85)

    # 1. Load Tokenizer & Shared GPT-2 Matrix Memory Model
    actual_model_path = model_name if os.path.exists(model_name) else "izzulgod/gpt2-indo-instruct-tuned"
    tokenizer = AutoTokenizer.from_pretrained(actual_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Memuat Unified GPT-2 Matrix Memory Model...")
    scaling = "none"
    use_lora = True
    lora_rank = 16
    adapter_data = None
    if os.path.exists(checkpoint_path):
        adapter_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = adapter_data.get("config", {})
        scaling = cfg.get("scaling", "none")
        use_lora = cfg.get("use_lora", any("lora" in k for k in adapter_data.get("adapter_state_dict", {})))
        lora_rank = cfg.get("lora_rank", 16)
    else:
        raise FileNotFoundError(f"Checkpoint adapter tidak ditemukan di {checkpoint_path}")

    model = GPT2MatrixMemoryModel(
        model_name_or_path=actual_model_path,
        capacity=128,
        scaling=scaling,
        freeze_backbone=True,
        use_lora=use_lora,
        lora_rank=lora_rank,
    ).to(device)
    model.load_adapter(adapter_data)
    print("✓ Sukses memuat bobot adapter.")

    # 2. Load Semantic Extractor & Shared BERTScore Evaluator
    print("Memuat IndoBERT Semantic Extractor...")
    extractor = SemanticSentenceExtractor("indobert", device=device)
    model.set_semantic_extractor(extractor)
    model.eval()

    bert_evaluator = BertScoreEvaluator(
        bert_model=extractor.encoder,
        bert_tokenizer=extractor.tokenizer,
        device=device,
    )

    # 3. Load Dataset
    samples = load_dataset(test_file, max_samples=num_samples)
    print(f"✓ Berhasil memuat {len(samples)} sampel dialog dari dataset test.\n")

    models_to_test = [
        ("Pure GPT-2 (Zero-shot)", "pure"),
        ("Full-Context GPT-2", "full_context"),
        ("MemoryBank W_q (scaling='none')", "mem_none"),
        ("MemoryBank W_q (scaling='sqrt')", "mem_sqrt"),
    ]

    metrics_store = {
        m_name: {
            "hit1": [] if "MemoryBank" in m_name else None,
            "hit3": [] if "MemoryBank" in m_name else None,
            "mrr": [] if "MemoryBank" in m_name else None,
            "entity_recall": [],
            "entity_token_recall": [],
            "bert_score_f1": [],
            "sentence_cosine": [],
            "input_tokens": [],
            "latency_ms": [],
        }
        for m_name, _ in models_to_test
    }

    print("Menjalankan inferensi dan penghitungan metrik...")
    for idx, sample in enumerate(samples, 1):
        turns = sample["turns"]
        target = sample["target_recall"]
        q_turn = target["query_turn"]
        q_text = target["question"]
        gt_entity = target["ground_truth"]
        gt_answer = target["answer"]

        # Tentukan target turn slot (posisi fakta disimpan)
        target_key = target.get("target_key")
        fact_turn = None
        for f in sample.get("facts", []):
            if f.get("key") == target_key:
                fact_turn = f.get("turn")
                break
        if fact_turn is None:
            fact_turn = 0  # fallback

        # Siapkan riwayat sebelum turn pertanyaan
        history_turns = turns[:q_turn]

        # -----------------------------------------------------------------
        # Model 1: Pure GPT-2 (Zero-shot, prompt hanya pertanyaan saat ini)
        # -----------------------------------------------------------------
        prompt_pure = f"User: {q_text}\nAI:"
        enc_pure = tokenizer(prompt_pure, return_tensors="pt").to(device)
        n_in_pure = enc_pure["input_ids"].shape[1]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_pure = model.gpt2.generate(
                enc_pure["input_ids"],
                attention_mask=enc_pure.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_pure = (time.perf_counter() - t0) * 1000.0
        gen_pure = tokenizer.decode(out_pure[0][n_in_pure:], skip_special_tokens=True).strip()

        # -----------------------------------------------------------------
        # Model 2: Full-Context GPT-2 (Seluruh riwayat dialog dimasukkan prompt)
        # -----------------------------------------------------------------
        fc_lines = []
        for t in history_turns:
            prefix = "User:" if t["role"] == "user" else "AI:"
            fc_lines.append(f"{prefix} {t['content']}")
        fc_lines.append(f"User: {q_text}\nAI:")
        prompt_fc = "\n".join(fc_lines)

        enc_fc = tokenizer(prompt_fc, return_tensors="pt").to(device)
        # Handle GPT-2 context limit (1024 tokens)
        if enc_fc["input_ids"].shape[1] > 1024 - max_new_tokens:
            enc_fc["input_ids"] = enc_fc["input_ids"][:, -(1024 - max_new_tokens):]
            if "attention_mask" in enc_fc:
                enc_fc["attention_mask"] = enc_fc["attention_mask"][:, -(1024 - max_new_tokens):]
        n_in_fc = enc_fc["input_ids"].shape[1]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_fc = model.gpt2.generate(
                enc_fc["input_ids"],
                attention_mask=enc_fc.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_fc = (time.perf_counter() - t0) * 1000.0
        gen_fc = tokenizer.decode(out_fc[0][n_in_fc:], skip_special_tokens=True).strip()

        # -----------------------------------------------------------------
        # Model 3 & 4: MemoryBank (W_q) dengan scaling='dim' dan 'sqrt'
        # -----------------------------------------------------------------
        for m_name, sc_mode, sc_val in [
            ("MemoryBank W_q (scaling='none')", "none", 1.0),
            ("MemoryBank W_q (scaling='sqrt')", "sqrt", 1.0 / math.sqrt(768.0)),
        ]:
            model.matrix_bank.scaling = sc_mode
            model.matrix_bank.scale_factor = sc_val
            model.reset_memory()
            # Tulis percakapan ke memory matrix slot
            with torch.no_grad():
                for t in history_turns:
                    entry = f"{'User' if t['role']=='user' else 'AI'}: {t['content']}"
                    vec = extractor.encode(entry, normalize=True)
                    model.matrix_bank.write(vec)

            # Hitung Retrieval Accuracy (Hit@1, Hit@3, MRR)
            q_sem = extractor.encode(q_text, normalize=True).to(device)
            q_proj = model.query_encoder(q_sem)
            _, attn = model.matrix_bank.read(q_proj)
            probs = attn[0, : model.matrix_bank.num_memories]
            sorted_indices = torch.argsort(probs, descending=True).tolist()

            top1_slot = sorted_indices[0] if sorted_indices else -1
            top3_slots = sorted_indices[:3]

            is_hit1 = 100.0 if top1_slot == fact_turn else 0.0
            is_hit3 = 100.0 if fact_turn in top3_slots else 0.0
            rank = sorted_indices.index(fact_turn) if fact_turn in sorted_indices else 999
            reciprocal_rank = 1.0 / (rank + 1)

            metrics_store[m_name]["hit1"].append(is_hit1)
            metrics_store[m_name]["hit3"].append(is_hit3)
            metrics_store[m_name]["mrr"].append(reciprocal_rank)

            # Generate respon dengan MemoryBank
            prompt_mem = f"User: {q_text}\nAI:"
            enc_mem = tokenizer(prompt_mem, return_tensors="pt").to(device)
            n_in_mem = enc_mem["input_ids"].shape[1]

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out_mem = model.generate(
                    input_ids=enc_mem["input_ids"],
                    query_text=q_text,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    top_k=20,
                    stop_token_ids=[tokenizer.eos_token_id],
                    use_memory=True,
                    write_after_gen=False,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat_mem = (time.perf_counter() - t0) * 1000.0
            gen_mem = tokenizer.decode(out_mem[0][n_in_mem:], skip_special_tokens=True).strip()

            # Record common metrics
            ent_bin, ent_tok = compute_entity_recall(gen_mem, gt_entity)
            _, _, bs_f1 = bert_evaluator.compute_bert_score(gen_mem, gt_answer)
            cos_sim = bert_evaluator.compute_sentence_cosine(gen_mem, gt_answer)

            metrics_store[m_name]["entity_recall"].append(ent_bin)
            metrics_store[m_name]["entity_token_recall"].append(ent_tok)
            metrics_store[m_name]["bert_score_f1"].append(bs_f1)
            metrics_store[m_name]["sentence_cosine"].append(cos_sim)
            metrics_store[m_name]["input_tokens"].append(n_in_mem)
            metrics_store[m_name]["latency_ms"].append(lat_mem)

        # Record metrics for Pure & Full Context
        ent_bin_p, ent_tok_p = compute_entity_recall(gen_pure, gt_entity)
        _, _, bs_f1_p = bert_evaluator.compute_bert_score(gen_pure, gt_answer)
        cos_p = bert_evaluator.compute_sentence_cosine(gen_pure, gt_answer)
        metrics_store["Pure GPT-2 (Zero-shot)"]["entity_recall"].append(ent_bin_p)
        metrics_store["Pure GPT-2 (Zero-shot)"]["entity_token_recall"].append(ent_tok_p)
        metrics_store["Pure GPT-2 (Zero-shot)"]["bert_score_f1"].append(bs_f1_p)
        metrics_store["Pure GPT-2 (Zero-shot)"]["sentence_cosine"].append(cos_p)
        metrics_store["Pure GPT-2 (Zero-shot)"]["input_tokens"].append(n_in_pure)
        metrics_store["Pure GPT-2 (Zero-shot)"]["latency_ms"].append(lat_pure)

        ent_bin_fc, ent_tok_fc = compute_entity_recall(gen_fc, gt_entity)
        _, _, bs_f1_fc = bert_evaluator.compute_bert_score(gen_fc, gt_answer)
        cos_fc = bert_evaluator.compute_sentence_cosine(gen_fc, gt_answer)
        metrics_store["Full-Context GPT-2"]["entity_recall"].append(ent_bin_fc)
        metrics_store["Full-Context GPT-2"]["entity_token_recall"].append(ent_tok_fc)
        metrics_store["Full-Context GPT-2"]["bert_score_f1"].append(bs_f1_fc)
        metrics_store["Full-Context GPT-2"]["sentence_cosine"].append(cos_fc)
        metrics_store["Full-Context GPT-2"]["input_tokens"].append(n_in_fc)
        metrics_store["Full-Context GPT-2"]["latency_ms"].append(lat_fc)

        if idx % 10 == 0 or idx == len(samples):
            print(f"  [Progress] Selesai memproses {idx}/{len(samples)} sampel dialog.")

    # 5. Summarize Results into Table
    summary = {}
    for m_name, _ in models_to_test:
        data = metrics_store[m_name]
        summary[m_name] = {
            "hit1": f"{sum(data['hit1']) / len(data['hit1']):.2f}%" if data["hit1"] is not None else "-",
            "hit3": f"{sum(data['hit3']) / len(data['hit3']):.2f}%" if data["hit3"] is not None else "-",
            "mrr": f"{sum(data['mrr']) / len(data['mrr']):.4f}" if data["mrr"] is not None else "-",
            "entity_recall": f"{sum(data['entity_recall']) / len(data['entity_recall']):.2f}%",
            "entity_token_recall": f"{sum(data['entity_token_recall']) / len(data['entity_token_recall']):.2f}%",
            "bert_score_f1": f"{sum(data['bert_score_f1']) / len(data['bert_score_f1']):.2f}%",
            "sentence_cosine": f"{sum(data['sentence_cosine']) / len(data['sentence_cosine']):.2f}%",
            "avg_input_tokens": f"{sum(data['input_tokens']) / len(data['input_tokens']):.1f}",
            "latency_ms": f"{sum(data['latency_ms']) / len(data['latency_ms']):.1f} ms",
        }

    # 6. Format Markdown Table
    md_table = [
        "| Model / Pendekatan | Hit@1 Retrieval | Entity Recall | BERTScore F1 | Sentence Cosine | Avg. Input Tokens | Latency (ms/turn) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, _ in models_to_test:
        s = summary[m_name]
        md_table.append(
            f"| **{m_name}** | {s['hit1']} | {s['entity_recall']} | {s['bert_score_f1']} | {s['sentence_cosine']} | {s['avg_input_tokens']} | {s['latency_ms']} |"
        )
    md_table_str = "\n".join(md_table)

    print("\n" + "=" * 85)
    print(" HASIL BENCHMARK MATRIKS PERBANDINGAN LENGKAP")
    print("=" * 85)
    print(md_table_str)
    print("=" * 85)

    # 7. Save to JSON and Markdown File
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "paper_metrics_evaluation.json")
    md_path = os.path.join(output_dir, "paper_metrics_table.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "num_samples": len(samples)}, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Rekomendasi Format Perbandingan untuk Laporan / Paper\n\n")
        f.write(f"Evaluasi dilakukan pada {len(samples)} sampel dialog percakapan test.\n\n")
        f.write(md_table_str)
        hit3_none = summary["MemoryBank W_q (scaling='none')"]["hit3"]
        hit3_sqrt = summary["MemoryBank W_q (scaling='sqrt')"]["hit3"]
        mrr_none = summary["MemoryBank W_q (scaling='none')"]["mrr"]
        mrr_sqrt = summary["MemoryBank W_q (scaling='sqrt')"]["mrr"]
        ent_pure = summary["Pure GPT-2 (Zero-shot)"]["entity_token_recall"]
        ent_fc = summary["Full-Context GPT-2"]["entity_token_recall"]
        ent_none = summary["MemoryBank W_q (scaling='none')"]["entity_token_recall"]

        f.write("\n\n### Detail Metrik Tambahan (Retrieval & Factual Alignment)\n\n")
        f.write(f"- **Hit@3 Retrieval**: `none`={hit3_none}, `sqrt`={hit3_sqrt}\n")
        f.write(f"- **MRR**: `none`={mrr_none}, `sqrt`={mrr_sqrt}\n")
        f.write(f"- **Entity Token Overlap**: `Pure`={ent_pure}, `Full-Context`={ent_fc}, `MemoryBank(none)`={ent_none}\n")

    print(f"\n✓ Hasil evaluasi disimpan ke:")
    print(f"   -> JSON     : {json_path}")
    print(f"   -> Markdown : {md_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark Paper Metrics for MemoryBank vs Baselines")
    parser.add_argument("--test_file", type=str, default="dataset/conversations_test.jsonl")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/matrix_adapter_best.pt")
    parser.add_argument("--model_name", type=str, default="gpt2-indo-instruct-tuned")
    parser.add_argument("--num_samples", type=int, default=30, help="Jumlah sampel untuk diuji (default: 30)")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    run_benchmark_pipeline(
        test_file=args.test_file,
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device_str=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
