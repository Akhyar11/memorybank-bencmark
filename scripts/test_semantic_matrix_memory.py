"""
scripts/test_semantic_matrix_memory.py
======================================
Pengujian Komparasi Ilmiah Pendekatan 1:
Last-Token Pooling (Baseline Lama) vs Semantic Vector Extractor (Baru).

Menguji:
1. Resolusi Retrieval Semantik (Fakta Target vs Distraktor).
2. Kemampuan Memory Matrix membedakan konteks esensial (Bebas bias kata terakhir).
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.gpt2_matrix_memory_model import GPT2MatrixMemoryModel
from models.semantic_extractor import SemanticSentenceExtractor
from models.seed import set_seed


TEST_CASES = [
    {
        "id": "CASE_1_PROFESI",
        "target_fact": "Btw pekerjaanku sehari-hari adalah Penjinak Naga Antariksa di stasiun luar angkasa.",
        "distractors": [
            "Halo selamat pagi, senang bisa mengobrol denganmu hari ini.",
            "Cuaca di luar hari ini cerah sekali dengan langit biru.",
            "Tadi siang aku makan nasi goreng sate yang sangat lezat.",
        ],
        "query": "Kamu masih ingat apa profesi atau pekerjaanku sehari-hari?",
        "keyword": "Penjinak Naga Antariksa",
    },
    {
        "id": "CASE_2_KOTA",
        "target_fact": "Aku baru saja membeli rumah baru dan pindah domisili ke Kota Atlantis Bawah Laut.",
        "distractors": [
            "Tadi aku membaca buku fiksi ilmiah tentang penjelajahan galaksi.",
            "Bagaimana cara terbaik untuk merawat kucing anggora di rumah?",
            "Aku sedang mempertimbangkan untuk belajar bahasa Jepang tahun ini.",
        ],
        "query": "Bisa ingatkan aku, ke kota mana aku baru saja pindah domisili?",
        "keyword": "Kota Atlantis Bawah Laut",
    },
    {
        "id": "CASE_3_MINUMAN",
        "target_fact": "Minuman unik yang selalu kuminum saat santai adalah Jus Baterai Lithium Dingin.",
        "distractors": [
            "Apakah kamu tahu resep membuat martabak manis yang lembut?",
            "Musim hujan seperti ini enaknya mendengarkan musik akustik.",
            "Pekerjaan kantor hari ini cukup padat dan melelahkan.",
        ],
        "query": "Minuman unik apa yang kuminum saat santai tadi?",
        "keyword": "Jus Baterai Lithium",
    },
]


def run_test():
    print("=" * 80)
    print("      EVALUASI SEMANTIC VECTOR EXTRACTOR (PENDEKATAN 1)")
    print("      Membandingkan Last-Token Pooling vs Semantic Extractor")
    print("=" * 80)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_dir = "gpt2-indo-instruct-tuned" if os.path.exists("gpt2-indo-instruct-tuned") else "izzulgod/gpt2-indo-instruct-tuned"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Inisialisasi Model Matrix Memory
    print("\n[1/3] Memuat GPT-2 Matrix Memory...")
    model = GPT2MatrixMemoryModel(
        model_name_or_path=model_dir,
        capacity=128,
        scaling="dim",
        freeze_backbone=True,
    ).to(device)

    ckpt_path = "checkpoints/gpt2_matrix_memory_best.pt"
    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = st["model_state_dict"] if "model_state_dict" in st else st
        model.load_state_dict(sd, strict=False)
        print(f"✓ Checkpoint loaded: {ckpt_path}")
    model.eval()

    # 2. Inisialisasi Semantic Sentence Extractor (IndoBERT)
    print("\n[2/3] Memuat SemanticSentenceExtractor (IndoBERT)...")
    extractor = SemanticSentenceExtractor(extractor_type="indobert", device=device)
    model.set_semantic_extractor(extractor)
    print("✓ SemanticSentenceExtractor siap.")

    # 3. Jalankan Pengujian Resolusi Retrieval
    print("\n[3/3] Menjalankan Pengujian Resolusi Retrieval Memori...")
    print("-" * 80)

    last_token_top1_count = 0
    semantic_top1_count = 0

    for idx, case in enumerate(TEST_CASES, 1):
        c_id = case["id"]
        target = case["target_fact"]
        distractors = case["distractors"]
        query = case["query"]
        all_turns = distractors[:1] + [target] + distractors[1:]  # target di tengah (slot index 1)
        target_slot = 1

        print(f"\n[Kasus {idx}] ID: {c_id}")
        print(f"  Fakta Inti    : \"{target}\" (Slot index: {target_slot})")
        print(f"  Pertanyaan    : \"{query}\"")

        # --- A. METODE LAMA: Last-Token Pooling ---
        model.reset_memory()
        with torch.no_grad():
            for t_text in all_turns:
                enc = tokenizer(f"User: {t_text}\n", return_tensors="pt").to(device)
                out = model.gpt2.transformer(enc["input_ids"])
                h_last = out.last_hidden_state[:, -1, :]  # Last token
                model.matrix_bank.write(h_last)

            # Query dengan prompt query last token
            enc_q = tokenizer(f"User: {query}\nAI:", return_tensors="pt").to(device)
            out_q = model.gpt2.transformer(enc_q["input_ids"])
            q_last = model.query_encoder(out_q.last_hidden_state[:, -1, :])
            _, act_old = model.matrix_bank.read(q_last)
            scores_old = act_old[0, :len(all_turns)].tolist()
            best_slot_old = int(torch.argmax(act_old[0, :len(all_turns)]).item())

        # --- B. METODE BARU: Semantic Vector Extractor ---
        model.reset_memory()
        with torch.no_grad():
            for t_text in all_turns:
                # Encode seluruh kalimat dengan Semantic Sentence Extractor
                v_sem = extractor.encode(t_text, normalize=True)
                model.matrix_bank.write(v_sem)

            # Query langsung dari teks pertanyaan dengan Semantic Extractor
            q_sem = extractor.encode(query, normalize=True)
            _, act_new = model.matrix_bank.read(q_sem)
            scores_new = act_new[0, :len(all_turns)].tolist()
            best_slot_new = int(torch.argmax(act_new[0, :len(all_turns)]).item())

        hit_old = (best_slot_old == target_slot)
        hit_new = (best_slot_new == target_slot)

        if hit_old:
            last_token_top1_count += 1
        if hit_new:
            semantic_top1_count += 1

        print(f"  Skor Aktivasi Memori Slot 0..{len(all_turns)-1}:")
        print(f"    - Last-Token Pooling  : {[round(s, 4) for s in scores_old]} -> Slot Pilihan: {best_slot_old} ({'✓ TEPAT' if hit_old else '✗ SALAH'})")
        print(f"    - Semantic Extractor  : {[round(s, 4) for s in scores_new]} -> Slot Pilihan: {best_slot_new} ({'✓ TEPAT' if hit_new else '✗ SALAH'})")

    total = len(TEST_CASES)
    print("\n" + "=" * 80)
    print("               HASIL EVALUASI RETRIEVAL SEMANTIK")
    print("=" * 80)
    print(f"Metode Ekstraksi Memori         | Akurasi Top-1 Target Retrieval (%)")
    print("-" * 80)
    print(f"Last-Token Pooling (Baseline)  | {last_token_top1_count / total * 100.0:30.1f}%")
    print(f"Semantic Extractor (Baru)      | {semantic_top1_count / total * 100.0:30.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    run_test()
