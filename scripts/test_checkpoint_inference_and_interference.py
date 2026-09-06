"""
scripts/test_checkpoint_inference_and_interference.py
======================================================
Comprehensive Inference & Memory Interference Benchmark
for GPT-2 + Turn-Level Semantic Memory Bank checkpoint.
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import TinyMemoryConfig


def load_model_from_checkpoint(checkpoint_path: str, model_dir: str, device: torch.device):
    print(f"Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_config = ckpt.get("memory_config", {})
    mem_config = TinyMemoryConfig(
        memory_dim=ckpt_config.get("memory_dim", 768),
        hidden_size=ckpt_config.get("hidden_size", 768),
        memory_capacity=ckpt_config.get("memory_capacity", 128),
        top_k=ckpt_config.get("top_k", 1),
        eviction_threshold_ratio=ckpt_config.get("eviction_threshold_ratio", 0.05),
        min_age_for_eviction=ckpt_config.get("min_age_for_eviction", 3),
        temperature=ckpt_config.get("temperature", 1.0),
    )

    model = GPT2MemoryModel(
        model_name_or_path=model_dir,
        memory_config=mem_config,
        freeze_backbone=True,
    ).to(device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "adapter_state_dict" in ckpt:
        model.load_state_dict(ckpt["adapter_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    print("Checkpoint successfully loaded and model set to eval() mode.")
    return model, tokenizer, ckpt


def test_single_turn_inference(model, tokenizer, device):
    print("\n" + "=" * 76)
    print(" 1. UJI INFERENSI DASAR (SINGLE-TURN TEXT GENERATION)")
    print("=" * 76)

    test_prompts = [
        "User: Siapa presiden pertama Republik Indonesia?\nAI:",
        "User: Apa ibukota dari negara Perancis?\nAI:",
        "User: Jelaskan secara singkat apa itu Python dalam pemrograman.\nAI:",
    ]

    for p in test_prompts:
        model.reset_memory()
        inputs = tokenizer(p, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=40,
                temperature=0.3,
                top_k=30,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = (time.perf_counter() - t0) * 1000.0
        generated_tokens = outputs[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        if "\n" in response:
            response = response.split("\n")[0].strip()

        print(f"Prompt : {p.strip()}")
        print(f"Respon : {response}")
        print(f"Tokens : {len(generated_tokens)} tokens ({elapsed:.1f} ms, {elapsed/max(1, len(generated_tokens)):.2f} ms/token)")
        print("-" * 76)


def test_multiturn_conversational_recall(model, tokenizer, device):
    print("\n" + "=" * 76)
    print(" 2. UJI INFERENSI MULTI-TURN & EPISODIC MEMORY RECALL")
    print("=" * 76)

    # We compare With Memory Bank vs Without Memory Bank
    conversation_turns = [
        "Halo, perkenalkan nama saya Akhyar dan saya bekerja sebagai AI Research Engineer di Bandung.",
        "Cuaca hari ini cukup cerah dan berawan.",
        "Saya sedang mengerjakan riset optimasi arsitektur neural memory bank.",
    ]
    recall_question = "Siapa nama saya, apa pekerjaan saya, dan di kota mana saya bekerja?"

    print(f"Skenario Percakapan (3 Turn):")
    for idx, t in enumerate(conversation_turns, 1):
        print(f"  Turn {idx}: {t}")
    print(f"Kueri Uji Recall: \"{recall_question}\"\n")

    for use_mem, label in [(True, "DENGAN MEMORY BANK (CHECKPOINT)"), (False, "TANPA MEMORY BANK (BASELINE ZERO-SHOT)")]:
        model.reset_memory()
        print(f"--- Mode: {label} ---")

        # Simulate earlier conversation turns
        if use_mem:
            for t in conversation_turns:
                prompt_t = f"User: {t}\nAI:"
                inp_t = tokenizer(prompt_t, return_tensors="pt").to(device)
                with torch.no_grad():
                    _ = model.generate(
                        input_ids=inp_t["input_ids"],
                        max_new_tokens=15,
                        temperature=0.3,
                        use_memory=True,
                    )

        # Query recall
        query_prompt = f"User: {recall_question}\nAI:"
        inp_q = tokenizer(query_prompt, return_tensors="pt").to(device)
        q_len = inp_q["input_ids"].shape[1]

        with torch.no_grad():
            out_q = model.generate(
                input_ids=inp_q["input_ids"],
                max_new_tokens=45,
                temperature=0.2,
                top_k=20,
                repetition_penalty=1.15,
                use_memory=use_mem,
            )

        resp_q = tokenizer.decode(out_q[0][q_len:], skip_special_tokens=True).strip()
        if "\n" in resp_q:
            resp_q = resp_q.split("\n")[0].strip()

        print(f"Slot Memori Terisi: {model.bank.num_memories}")
        print(f"Respon Model     : {resp_q}\n")


def test_memory_interference_and_distractors(model, tokenizer, device):
    print("\n" + "=" * 76)
    print(" 3. UJI KETAHANAN INTERFERENSI & DISTRAKTOR (INTERFERENCE TEST)")
    print("=" * 76)
    print("Tujuan: Menguji apakah representasi memori target mampu bertahan")
    print("dari gangguan distraktor/interferensi topik lain dalam memori bank.\n")

    # Target Fact
    target_fact = "Fakta penting: Proyek bernama AlphaCode dikembangkan oleh tim DeepMind pada tahun 2022."
    recall_query = "Siapa tim pengembang proyek AlphaCode dan tahun berapa?"

    # Distractor pools
    distractor_pool = [
        "Resep nasi goreng spesial memerlukan bawang merah, kecap manis, dan telur ayam.",
        "Menara Eiffel diresmikan di Paris pada tahun 1889 untuk pameran dunia.",
        "Timnas Argentina menjuarai Piala Dunia FIFA tahun 2022 di Qatar.",
        "Gunung Fuji adalah gunung berapi tertinggi di Jepang dengan ketinggian 3.776 meter.",
        "Kucing persia memerlukan perawatan bulu teratur agar tidak kusut dan rontok.",
        "Bumi mengelilingi matahari dalam waktu sekitar 365,25 hari.",
        "Candi Borobudur dibangun pada abad ke-8 masehi oleh wangsa Syailendra.",
        "Studi fisika kuantum mempelajari perilaku partikel subatomik pada skala mikroskopis.",
        "Kopi arabika memiliki rasa lebih asam dan aroma buah dibandingkan kopi robusta.",
        "Danau Toba terbentuk akibat letusan supervolcano puluhan ribu tahun lalu.",
        "Minyak kelapa sawit adalah salah satu komoditas ekspor terbesar Indonesia.",
        "Mobil listrik menggunakan motor penggerak baterai lithium-ion.",
    ]

    interference_levels = [0, 2, 5, 10, 12]

    for num_dist in interference_levels:
        model.reset_memory()

        # 1. Simpan Target Fact ke Memory Bank
        enc_target = tokenizer(f"User: {target_fact}\nAI:", return_tensors="pt").to(device)
        with torch.no_grad():
            out_target = model.gpt2.transformer(input_ids=enc_target["input_ids"], return_dict=True)
            h_target = out_target.last_hidden_state[:, -1, :].squeeze(0)  # (768,)
            model.bank.add_memory(h_target)

        # 2. Injeksi Distraktor (Interferensi)
        for i in range(num_dist):
            dist_text = distractor_pool[i]
            enc_d = tokenizer(f"User: {dist_text}\nAI:", return_tensors="pt").to(device)
            with torch.no_grad():
                out_d = model.gpt2.transformer(input_ids=enc_d["input_ids"], return_dict=True)
                h_d = out_d.last_hidden_state[:, -1, :].squeeze(0)
                model.bank.add_memory(h_d)

        # 3. Kueri Retrieval
        enc_q = tokenizer(f"User: {recall_query}\nAI:", return_tensors="pt").to(device)
        with torch.no_grad():
            out_q = model.gpt2.transformer(input_ids=enc_q["input_ids"], return_dict=True)
            h_q = out_q.last_hidden_state[:, -1, :]  # (1, 768)
            C_q = model.c_proj(h_q)                  # (1, 768)

            # Hitung cosine similarity ke seluruh memori di bank
            mem_tensor = torch.stack([m.to(device) for m in model.bank.memories], dim=0)  # (M, 768)
            C_norm = F.normalize(C_q, p=2, dim=-1)
            mem_norm = F.normalize(mem_tensor, p=2, dim=-1)
            similarities = torch.matmul(C_norm, mem_norm.t()).squeeze(0)  # (M,)

            target_sim = similarities[0].item()
            best_idx = torch.argmax(similarities).item()
            best_sim = similarities[best_idx].item()

            # Generate jawaban
            q_len = enc_q["input_ids"].shape[1]
            gen_out = model.generate(
                input_ids=enc_q["input_ids"],
                max_new_tokens=35,
                temperature=0.2,
                top_k=20,
                repetition_penalty=1.15,
                use_memory=True,
            )
            ans = tokenizer.decode(gen_out[0][q_len:], skip_special_tokens=True).strip()
            if "\n" in ans:
                ans = ans.split("\n")[0].strip()

        rank_target = (torch.argsort(similarities, descending=True) == 0).nonzero().item() + 1
        status = "PASSED" if rank_target == 1 else "DISTRACTED"

        print(f"[Tingkat Interferensi]: {num_dist:2d} Distraktor (Total Slot Memori: {model.bank.num_memories})")
        print(f"  └─ Rank Target Memory : #{rank_target} (Sim: {target_sim:.4f} | Top Sim: {best_sim:.4f})")
        print(f"  └─ Status Retrieval   : {status}")
        print(f"  └─ Respon Jawaban     : {ans}")
        print("-" * 76)


def test_held_out_dataset_metrics(model, tokenizer, device, num_samples=15):
    print("\n" + "=" * 76)
    print(f" 4. EVALUASI METRIK DATASET UJI HELD-OUT ({num_samples} SAMPEL DIALOG)")
    print("=" * 76)

    test_file = os.path.join(PROJECT_ROOT, "dataset", "conversations_test.jsonl")
    if not os.path.exists(test_file):
        print(f"Dataset {test_file} tidak ditemukan, melewati tahap ini.")
        return

    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= num_samples:
                break

    print(f"Menguji {len(samples)} dialog dari {test_file}...")

    methods = [("Tanpa Memory (Baseline)", False), ("Causal Memory Bank (Checkpoint)", True)]
    results = {m_name: {"em": [], "f1": [], "latencies": []} for m_name, _ in methods}

    for item in samples:
        facts = item.get("facts", [])
        turns = item.get("turns", [])
        target_recall = item.get("target_recall", {})
        recall_q = target_recall.get("question", "Siapa namaku?")
        recall_a = target_recall.get("answer", "")
        ground_truth = target_recall.get("ground_truth", "")

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

        for m_name, use_mem in methods:
            model.reset_memory()

            if use_mem:
                with torch.no_grad():
                    for u_text, ai_text in fact_turns:
                        w_enc = tokenizer(f"User: {u_text}\nAI: {ai_text}\n", max_length=128, truncation=True, return_tensors="pt").to(device)
                        model(w_enc["input_ids"], use_memory=True, persist_memory=True)

            t0 = time.perf_counter()
            with torch.no_grad():
                out_gen = model.generate(
                    input_ids=enc_query["input_ids"],
                    max_new_tokens=25,
                    temperature=0.1,
                    top_k=20,
                    use_memory=use_mem,
                )
            t1 = time.perf_counter()

            gen_tokens = out_gen[0][q_len:]
            num_tokens = max(1, len(gen_tokens))
            latency = ((t1 - t0) / num_tokens) * 1000.0

            prediction = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Exact Match
            em = 100.0 if ground_truth.lower().strip() in prediction.lower().strip() else 0.0

            # Token F1
            pred_toks = prediction.lower().split()
            gt_toks = recall_a.lower().split()
            common = set(pred_toks) & set(gt_toks)
            if common and pred_toks and gt_toks:
                p = len(common) / len(pred_toks)
                r = len(common) / len(gt_toks)
                f1 = (2 * p * r / (p + r)) * 100.0
            else:
                f1 = 0.0

            results[m_name]["em"].append(em)
            results[m_name]["f1"].append(f1)
            results[m_name]["latencies"].append(latency)

    print("\n" + "=" * 76)
    print(f"{'Metode':<35} | {'Exact Match':<12} | {'Token F1':<10} | {'Latency':<10}")
    print("-" * 76)
    for m_name, _ in methods:
        avg_em = sum(results[m_name]["em"]) / len(results[m_name]["em"])
        avg_f1 = sum(results[m_name]["f1"]) / len(results[m_name]["f1"])
        avg_lat = sum(results[m_name]["latencies"]) / len(results[m_name]["latencies"])
        print(f"{m_name:<35} | {avg_em:>10.2f}% | {avg_f1:>8.2f}% | {avg_lat:>7.1f} ms")
    print("=" * 76)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print(f" PENGUJIAN INFERENSI & INTERFERENSI MODEL CHECKPOINT")
    print(f" Perangkat: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 76)

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "gpt2_causal_memory_best.pt")
    model_dir = os.path.join(PROJECT_ROOT, "gpt2-indo-instruct-tuned")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint tidak ditemukan di {checkpoint_path}")
        return

    model, tokenizer, ckpt = load_model_from_checkpoint(checkpoint_path, model_dir, device)

    # 1. Single-turn inference test
    test_single_turn_inference(model, tokenizer, device)

    # 2. Multi-turn conversational episodic memory recall
    test_multiturn_conversational_recall(model, tokenizer, device)

    # 3. Interference & distractor resistance test
    test_memory_interference_and_distractors(model, tokenizer, device)

    # 4. Quantitative held-out evaluation
    test_held_out_dataset_metrics(model, tokenizer, device, num_samples=15)

    print("\n✓ Semua tahapan pengujian selesai dilakukan dengan sukses.")


if __name__ == "__main__":
    main()
