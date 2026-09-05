"""
tests/test_interference_benchmark.py
======================================
Interference & Distractor Resistance Benchmark untuk TinyMemoryBank & GPT-2.

Menguji ketahanan memori terhadap:
1. Low-Noise & High-Noise Distractors (apakah target memori tertimpa atau tetap utuh).
2. Orthogonal Distractors (injeksi 15-30 informasi lain di memori).
3. Conversational Distractor Test (Percakapan dengan banyak turn pengganggu).
"""
import os
import sys
sys.path.insert(0, "/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark")

import torch
import torch.nn.functional as F
from models.tiny_memory_bank import TinyMemoryBank, TinyMemoryConfig, STATE_ACTIVE
from models.gpt2_memory_model import GPT2MemoryModel
from transformers import AutoTokenizer


def run_vector_interference_benchmark():
    print("=" * 68)
    print("   [LEVEL 1]: VECTOR INTERFERENCE & DISTRACTOR BENCHMARK")
    print("=" * 68)

    cfg = TinyMemoryConfig(memory_capacity=128, memory_dim=64, hidden_size=64, memory_top_k=4)
    bank = TinyMemoryBank(cfg)
    # Untuk standalone benchmark arsitektur memori, selaraskan ruang embedding Q-K
    bank.q_proj.weight.data.copy_(bank.k_proj.weight.data)
    bank.reset_memory()

    # 1. Target Vector
    torch.manual_seed(42)
    dim = cfg.memory_dim
    h_target = torch.randn(1, dim)
    h_target = h_target / torch.norm(h_target)

    # Tulis Target ke Slot
    target_slot = bank.write(h_target, torch.ones(1), torch.ones(1)).item()
    print(f"✓ Target Fact disimpan di Slot #{target_slot}")

    # Baca awal (sebelum gangguan)
    v_read_before, scores_before = bank.read(h_target, return_scores=True)
    expected_v = bank.v_proj(h_target)
    sim_before = F.cosine_similarity(v_read_before, expected_v).item()
    top_slot_before = torch.argmax(scores_before[0]).item()
    print(f"  └─ Sim sebelum distractor: {sim_before:.4f} (Top Slot #{top_slot_before})")

    # 2. Injeksi Distractor Berbagai Tingkat Noise & Kuantitas
    test_cases = [
        ("Orthogonal Distractors (Informasi Topik Lain)", 15, None),
        ("Low-Noise Distractors (Informasi Serupa, noise=0.1)", 10, 0.1),
        ("Medium-Noise Distractors (Informasi Menyerupai, noise=0.3)", 15, 0.3),
        ("High-Noise Distractors (Informasi Menyerupai, noise=0.5)", 20, 0.5),
    ]

    for name, num_dist, noise in test_cases:
        # Reset dan tulis target
        bank.reset_memory()
        target_slot = bank.write(h_target, torch.ones(1), torch.ones(1)).item()

        # Injeksi distractors
        for d_idx in range(num_dist):
            if noise is None:
                # Distractor acak independen (ortogonal)
                d_vec = torch.randn(1, dim)
            else:
                # Distractor yang berinterferensi dekat target
                d_vec = h_target + torch.randn(1, dim) * noise
            d_vec = d_vec / torch.norm(d_vec)
            bank.write(d_vec, torch.ones(1), torch.ones(1))

        # Baca kembali target
        v_read_after, scores_after = bank.read(h_target, return_scores=True)
        sim_after = F.cosine_similarity(v_read_after, expected_v).item()
        top_slot_after = torch.argmax(scores_after[0]).item()
        active_count = bank.active_count

        status = "PASSED (Kuat)" if top_slot_after == target_slot and sim_after > 0.6 else "FAILED"
        print(f"\n[Test]: {name} ({num_dist} distractors)")
        print(f"  Slot aktif terisi     : {active_count}/{cfg.memory_capacity}")
        print(f"  Target retrieval rank : Slot #{top_slot_after} (Expected: #{target_slot})")
        print(f"  Cosine Similarity     : {sim_after:.4f} (Sebelum: {sim_before:.4f})")
        print(f"  Status Ketahanan      : {status}")
        assert top_slot_after == target_slot, f"Interference gagal: Target tergeser ke slot {top_slot_after}"

    print("\n✓ [LEVEL 1 BENCHMARK]: Seluruh pengujian interferensi vektor BERHASIL!\n")


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
        memory_read_threshold=0.2
    )
    model = GPT2MemoryModel(model_name_or_path=model_dir, memory_config=mem_config).to(device)

    # Muat checkpoint jika ada
    ckpt_path = "/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/checkpoints/best_adapter.pt"
    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(st, strict=False)
        print(f"✓ Memuat adapter checkpoint: {ckpt_path}")
    else:
        print("ℹ Checkpoint belum ada, menggunakan bobot adapter inisialisasi residual")

    model.eval()
    model.reset_memory()

    # 1. Tulis Fakta Target Penting (Konsep N=8 Token Representasi, Opsi 1)
    target_fact = "Halo, perkenalkan namaku Akhyar dan aku bekerja sebagai Programmer."
    enc_target = tok(f"User: {target_fact}\n", return_tensors="pt").to(device)
    pfx_ids = tok("User: ", return_tensors="pt")["input_ids"]
    pfx_len = pfx_ids.shape[1]

    with torch.no_grad():
        out_t = model(enc_target["input_ids"])
        h_target = out_t["hidden_states"][0, pfx_len:-1]
        wp_target = out_t["write_probs"][0, pfx_len:-1]
        target_slots = model.write_memory_chunked(h_target, wp_target, chunk_size=8).tolist()

    print(f"✓ Fakta Target: '{target_fact}'")
    print(f"  └─ Ditulis ke {len(target_slots)} Segmen Slot: #{target_slots} (Active slots: {model.bank.active_count})")

    # 2. Injeksi 8 Kalimat Percakapan Distractor (Topik Berbeda-beda)
    distractors = [
        "Hari ini cuaca di luar sangat panas dan terik.",
        "Aku baru saja memasak nasi goreng untuk sarapan pagi.",
        "Pesawat terbang menggunakan prinsip gaya angkat aerodinamika.",
        "Indonesia memiliki ribuan pulau dari Sabang sampai Merauke.",
        "Baterai lithium-ion banyak digunakan pada kendaraan listrik.",
        "Gunung Bromo terkenal dengan pemandangan matahari terbitnya yang indah.",
        "Kucing tidur rata-rata 12 sampai 16 jam dalam sehari.",
        "Kopi arabika memiliki rasa yang lebih asam dibanding robusta."
    ]

    print(f"\n[Menginjeksi {len(distractors)} percakapan distractor ke memori...]")
    for i, d in enumerate(distractors):
        enc_d = tok(f"User: {d}\n", return_tensors="pt").to(device)
        with torch.no_grad():
            out_d = model(enc_d["input_ids"])
            h_d = out_d["hidden_states"][0, pfx_len:-1]
            wp_d = out_d["write_probs"][0, pfx_len:-1]
            d_slots = model.write_memory_chunked(h_d, wp_d, chunk_size=8).tolist()
        print(f"  Distractor #{i+1:02d} -> {len(d_slots)} Slot ({d_slots[0] if d_slots else '-'}-{d_slots[-1] if d_slots else '-'}): '{d[:35]}...'")

    print(f"\nTotal slot aktif setelah {len(distractors)} distractor: {model.bank.active_count}/128")

    # 3. Recall Target Query
    recall_query = "Siapa namaku dan apa pekerjaanku?"
    enc_q = tok(f"User: {recall_query}\nAI:", return_tensors="pt").to(device)
    with torch.no_grad():
        q_len = enc_q["input_ids"].shape[1]
        out_gen = model.generate(
            input_ids=enc_q["input_ids"],
            max_new_tokens=25,
            temperature=0.3,
            stop_token_ids=[199]
        )
    ans = tok.decode(out_gen[0][q_len:], skip_special_tokens=True).strip()

    # Periksa perhatian retrieval
    with torch.no_grad():
        out_eval = model(enc_q["input_ids"])
        retrieval_scores = model.bank.last_scores[0]
        top_retrieved_slot = torch.argmax(retrieval_scores).item()

    print("\n[HASIL EVALUASI RECALL PASCA DISTRACTOR]:")
    print("-" * 68)
    print(f"Pertanyaan Recall       : {recall_query}")
    print(f"Slot yang diretrieve    : Slot #{top_retrieved_slot} (Target Slots: {target_slots})")
    print(f"Output Jawaban Model    : {ans}")

    if top_retrieved_slot in target_slots:
        print("✅ KESIMPULAN: TAHAN INTERFERENSI (Interference Resistant)!")
        print("   Slot target tidak tertimpa maupun terdistorsi oleh 8 percakapan pengganggu.")
    else:
        print("⚠️ Catatan: Slot target tergeser oleh distractor.")


if __name__ == "__main__":
    run_vector_interference_benchmark()
    run_conversational_interference_benchmark()
