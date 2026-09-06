"""
scripts/test_memory_distractor.py
=================================
Menguji ketahanan Memory Bank terhadap distraksi bertahap:
1. Berikan fakta utama di Turn 1.
2. Berikan 3 distraksi berurutan (Turn 2, Turn 3, Turn 4) dengan topik berbeda.
3. Ajukan kueri penarikan (recall) fakta utama di Turn 5.
4. Analisis perilaku memori:
   - Distribusi Cosine Similarity query ke tiap slot (0-7).
   - Efek User Prompt vs AI Response dalam Memory Bank.
   - Resistensi terhadap Recency Bias (apakah memori lama kalah oleh distraksi baru).
   - Generasi teks respon AI (Dengan Memory vs Tanpa Memory).
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import TinyMemoryConfig


def run_distractor_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Menggunakan: {device}")

    model_dir = "gpt2-indo-instruct-tuned"
    checkpoint_path = "checkpoints/gpt2_causal_memory_best.pt"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Checkpoint] Memuat {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = TinyMemoryConfig(
        memory_dim=768,
        hidden_size=768,
        memory_capacity=128,
        top_k=1,
        eviction_threshold_ratio=0.05,
        min_age_for_eviction=10,
        temperature=1.0,
    )
    model = GPT2MemoryModel(model_dir, memory_config=cfg).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "adapter_state_dict" in ckpt:
        model.load_state_dict(ckpt["adapter_state_dict"], strict=False)
    model.eval()

    # Skenario Percakapan
    turn1_u = "Halo, namaku Akhyar. Aku bekerja sebagai Machine Learning Engineer dan tinggal di Bandung."
    turn2_u = "Bisa jelaskan resep dan cara membuat nasi goreng spesial yang lezat?"
    turn3_u = "Kira-kira apa saja tempat wisata alam yang menarik untuk dikunjungi di Pulau Bali?"
    turn4_u = "Bagaimana prinsip kerja mesin mobil listrik dan perbedaannya dengan mesin bensin?"
    recall_q = "Ngomong-ngomong di luar obrolan tadi, kamu masih ingat siapa namaku, apa pekerjaanku, dan di mana aku tinggal?"

    turns = [
        ("Turn 1 (FAKTA UTAMA)", turn1_u),
        ("Turn 2 (DISTRAKSI 1: Kuliner)", turn2_u),
        ("Turn 3 (DISTRAKSI 2: Wisata)", turn3_u),
        ("Turn 4 (DISTRAKSI 3: Mobil Listrik)", turn4_u),
    ]

    model.reset_memory()
    slot_labels = []

    print("\n" + "=" * 80)
    print(" 1. PROSES PENYIMPANAN KE PERCAKAPAN (1 FAKTA + 3 DISTRAKSI)")
    print("=" * 80)

    for t_name, u_text in turns:
        prompt = f"User: {u_text}\nAI:"
        inp = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids=inp["input_ids"],
                max_new_tokens=30,
                temperature=0.3,
                top_k=20,
                use_memory=True,
            )
        resp = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if "\n" in resp:
            resp = resp.split("\n")[0].strip()

        slot_labels.append(f"Prompt {t_name}")
        slot_labels.append(f"AI Resp {t_name}")

        print(f"\n[{t_name}]")
        print(f"User : {u_text}")
        print(f"AI   : {resp}")
        print(f"-> Jumlah slot memori aktif sekarang: {model.bank.num_memories}")

    # Analisis Retrieval Sebelum Generate Kueri Recall
    print("\n" + "=" * 80)
    print(" 2. ANALISIS RETRIEVAL & PERILAKU MEMORY BANK (TURN 5 RECALL)")
    print("=" * 80)
    print(f"User Recall Query: \"{recall_q}\"")

    prompt_q = f"User: {recall_q}\nAI:"
    inp_q = tokenizer(prompt_q, return_tensors="pt").to(device)
    q_len = inp_q["input_ids"].shape[1]

    num_accumulated = model.bank.num_memories
    assert num_accumulated == 8, f"Expected 8 slots, got {num_accumulated}"

    with torch.no_grad():
        # Hitung hidden state kueri
        out_q_trans = model.gpt2.transformer(input_ids=inp_q["input_ids"], return_dict=True)
        h_q = out_q_trans.last_hidden_state[:, -1, :]  # (1, 768)
        C = model.c_proj(h_q)  # (1, 768)

        # Hitung cosine similarity ke tiap slot
        mem_tensor = torch.stack([m.to(device) for m in model.bank.memories], dim=0)  # (8, 768)
        C_norm = F.normalize(C, p=2, dim=-1)
        mem_norm = F.normalize(mem_tensor, p=2, dim=-1)
        sims = torch.matmul(C_norm, mem_norm.t()).squeeze(0)  # (8,)

        # Generasi DENGAN MEMORI
        out_with_mem = model.generate(
            input_ids=inp_q["input_ids"],
            max_new_tokens=40,
            temperature=0.2,
            top_k=20,
            repetition_penalty=1.15,
            use_memory=True,
        )

    resp_with_mem = tokenizer.decode(out_with_mem[0][q_len:], skip_special_tokens=True).strip()
    if "\n" in resp_with_mem:
        resp_with_mem = resp_with_mem.split("\n")[0].strip()

    # Generasi TANPA MEMORI (Baseline)
    model.reset_memory()
    with torch.no_grad():
        out_no_mem = model.generate(
            input_ids=inp_q["input_ids"],
            max_new_tokens=40,
            temperature=0.2,
            top_k=20,
            repetition_penalty=1.15,
            use_memory=False,
        )
    resp_no_mem = tokenizer.decode(out_no_mem[0][q_len:], skip_special_tokens=True).strip()
    if "\n" in resp_no_mem:
        resp_no_mem = resp_no_mem.split("\n")[0].strip()

    sorted_indices = torch.argsort(sims, descending=True).tolist()
    ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}

    print("\n--- TABEL SIMILARITAS RETRIEVAL MEMORI ---")
    print(f"{'Slot':<9} | {'Tipe Isi Slot':<37} | {'Cosine Sim':<12} | {'Rank':<8} | Keterangan")
    print("-" * 85)
    for idx in range(len(sims)):
        lbl = slot_labels[idx]
        sim_val = sims[idx].item()
        rank_val = ranks[idx]
        ket = "[TARGET FAKTA UTAMA]" if idx == 0 else ""
        print(f"Slot [{idx:02d}] | {lbl:<37} | {sim_val:>10.4f} | Rank #{rank_val:<3} | {ket}")
    print("-" * 85)

    winner_slot = sorted_indices[0]
    print(f"\n[HASIL RETRIEVAL]")
    print(f"- Pemenang Top-1 Retrieval : Slot [{winner_slot:02d}] ({slot_labels[winner_slot]})")
    print(f"- Cosine Similarity       : {sims[winner_slot].item():.4f}")
    print(f"- Gap ke Rank #2           : {sims[sorted_indices[0]].item() - sims[sorted_indices[1]].item():+.4f}")
    print(f"- Status Target Fakta      : {'SUKSES TERPILIH (RANK #1)' if winner_slot == 0 else 'GAGAL/TERDISTRAKSI'}")

    print("\n" + "=" * 80)
    print(" 3. PERBANDINGAN RESPON MODEL")
    print("=" * 80)
    print(f"Kueri User         : \"{recall_q}\"")
    print(f"Respon DENGAN MEMORY : \"{resp_with_mem}\"")
    print(f"Respon TANPA MEMORY  : \"{resp_no_mem}\"")
    print("=" * 80)


if __name__ == "__main__":
    run_distractor_experiment()
