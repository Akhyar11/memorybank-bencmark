"""
scripts/prepare_chatgpt_conversations.py
=========================================
Parser & Data Cleaner untuk Ekspor Percakapan ChatGPT (OpenAI ZIP / JSON).

Fitur Pembersihan:
  1. Auto-discovery berkas ZIP ekspor ChatGPT di ~/Unduhan / ~/Downloads.
  2. Rekonstruksi rantai percakapan aktif secara kronologis (current_node -> root).
  3. Pembersihan artefak internal (menghapus thoughts/reasoning tokens model o1/o3-mini).
  4. Pembersihan referensi multimodal (mengekstrak teks murni, membuang asset pointer gambar).
  5. Pembersihan sitasi OpenAI (【1†source】).
  6. Normalisasi alur percakapan:
     - Menggabungkan giliran berturut-turut dengan peran yang sama (user-user / ai-ai).
     - Menjamin dialog dimulai dari 'user' dan berakhiran 'assistant' (selang-seling genap).
  7. Filter kualitas dialog:
     - Minimal 4 giliran (min 2 pasang user-assistant) untuk melatih kesinambungan memori multi-turn.
     - Membuang pesan error sistem, timeout, atau respons kosong.
     - Pembatasan panjang karakter per giliran (mencegah OOM dari log/dump berukuran raksasa).
  8. Split deterministik (80% Train, 10% Val, 10% Test) dengan Universal Seeder.
"""

import os
import sys
import re
import json
import glob
import zipfile
import argparse
from typing import List, Dict, Any, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from models.seed import set_seed
except ImportError:
    import random
    def set_seed(s: int = 42):
        random.seed(s)


DEFAULT_ZIP_CANDIDATES = [
    "/home/akhyar/Unduhan/8766716822ea6bbc15133b9fdd644dee89186edc0d9502c9bc5d288d4efae226-2026-09-05-16-08-32-4dc7aafcc52a4a7482a7e83d419d581d.zip",
    "/kaggle/input/chatgpt-export/chatgpt_export.zip",
]

KNOWN_SYSTEM_ERROR_PATTERNS = [
    r"^An error occurred",
    r"^I apologize, but I encountered an error",
    r"^The previous model response was interrupted",
    r"^This content may violate our content policy",
    r"^Rate limit reached",
]


def find_latest_chatgpt_zip() -> Optional[str]:
    """Mencari berkas zip ChatGPT terbaru di folder unduhan jika kandidat tidak ditemukan."""
    for cand in DEFAULT_ZIP_CANDIDATES:
        if os.path.exists(cand):
            return cand

    search_dirs = [
        os.path.expanduser("~/Unduhan"),
        os.path.expanduser("~/Downloads"),
        os.getcwd(),
    ]
    for sdir in search_dirs:
        if os.path.isdir(sdir):
            zips = glob.glob(os.path.join(sdir, "*.zip"))
            zips.sort(key=os.path.getmtime, reverse=True)
            for z in zips:
                try:
                    with zipfile.ZipFile(z, "r") as test_z:
                        names = test_z.namelist()
                        if any("conversations-" in n or n == "conversations.json" for n in names):
                            return z
                except Exception:
                    continue
    return None


def clean_text_content(raw_text: str) -> str:
    """Membersihkan teks dari sitasi OpenAI, spasi berlebih, dan karakter non-standar."""
    # 1. Hapus sitasi sumber OpenAI: 【12†source】atau 【1:2†source】
    text = re.sub(r"【\d+(?::\d+)?†source】", "", raw_text)
    # 2. Normalisasi baris kosong ganda yang berlebihan (maksimal 2 baris baru berturut-turut)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 3. Hapus whitespace di tepi
    text = text.strip()
    return text


def is_system_error_response(text: str) -> bool:
    """Mendeteksi apakah pesan asisten merupakan pesan error sistem ChatGPT."""
    for pattern in KNOWN_SYSTEM_ERROR_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def extract_single_conversation(
    conv_data: Dict[str, Any],
    min_turns: int = 4,
    max_chars_per_turn: int = 3000,
    min_turn_chars: int = 4,
) -> Optional[Dict[str, Any]]:
    """
    Mengekstrak dan menormalisasi satu sesi percakapan dari struktur mapping ChatGPT:
    1. Backtrack dari current_node ke root.
    2. Abaikan thoughts/reasoning_recap dan bagian non-teks.
    3. Normalisasi giliran berturut-turut.
    4. Validasi kualitas dan panjang dialog.
    """
    mapping = conv_data.get("mapping", {})
    current_node = conv_data.get("current_node")
    if not mapping or not current_node:
        return None

    # 1. Telusuri rantai pesan dari leaf ke root
    curr = current_node
    raw_nodes = []
    while curr:
        node = mapping.get(curr)
        if not node:
            break
        msg = node.get("message")
        if msg:
            raw_nodes.append(msg)
        curr = node.get("parent")
    raw_nodes.reverse()

    # 2. Filter konten dan peran yang valid
    extracted_turns: List[Dict[str, str]] = []
    for msg in raw_nodes:
        author = msg.get("author", {})
        role = author.get("role")
        if role not in ("user", "assistant"):
            continue

        content = msg.get("content", {})
        ctype = content.get("content_type", "")
        # Abaikan internal scratchpad / thoughts model penalaran
        if ctype in ("thoughts", "reasoning_recap"):
            continue

        parts = content.get("parts", [])
        text_chunks = []
        for p in parts:
            if isinstance(p, str):
                text_chunks.append(p)
            elif isinstance(p, dict) and p.get("content_type") == "text":
                text_chunks.append(p.get("text", ""))

        clean_text = clean_text_content("".join(text_chunks))
        if not clean_text or len(clean_text) < min_turn_chars:
            continue

        # Periksa apakah asisten mengembalikan error sistem
        if role == "assistant" and is_system_error_response(clean_text):
            return None

        # Truncate secara anggun jika melebihi batas karakter maksimum
        if len(clean_text) > max_chars_per_turn:
            clean_text = clean_text[:max_chars_per_turn] + "\n... [potongan teks panjang]"

        extracted_turns.append({"role": role, "content": clean_text})

    if not extracted_turns:
        return None

    # 3. Normalisasi pergantian giliran (Strict Alternating: User -> Assistant -> User -> Assistant)
    normalized_turns: List[Dict[str, str]] = []
    for t in extracted_turns:
        if not normalized_turns:
            # Percakapan wajib dimulai oleh user
            if t["role"] == "user":
                normalized_turns.append(t)
        else:
            if t["role"] == normalized_turns[-1]["role"]:
                # Gabungkan pesan berturut-turut dari peran yang sama
                normalized_turns[-1]["content"] += "\n\n" + t["content"]
            else:
                normalized_turns.append(t)

    # Pastikan percakapan diakhiri oleh asisten (panjang genap)
    if len(normalized_turns) % 2 != 0:
        normalized_turns = normalized_turns[:-1]

    # 4. Validasi syarat minimal giliran
    if len(normalized_turns) < min_turns:
        return None

    return {
        "id": conv_data.get("id"),
        "title": conv_data.get("title", "Obrolan Tanpa Judul"),
        "create_time": conv_data.get("create_time"),
        "update_time": conv_data.get("update_time"),
        "num_turns": len(normalized_turns),
        "turns": normalized_turns,
    }


def parse_chatgpt_archive(
    zip_path: str,
    output_dir: str = "dataset",
    min_turns: int = 4,
    max_chars_per_turn: int = 3000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Mengekstrak, membersihkan, dan mengekspor percakapan ChatGPT menjadi JSONL siap latih."""
    set_seed(seed)
    import random

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Berkas ZIP ChatGPT tidak ditemukan di: {zip_path}")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 76)
    print("      PARSER & DATA CLEANER EKSPOR PERCAKAPAN CHATGPT UNTUK MEMORY BANK")
    print("=" * 76)
    print(f"  Arsip ZIP Sumber      : {os.path.abspath(zip_path)}")
    print(f"  Direktori Output      : {os.path.abspath(output_dir)}")
    print(f"  Syarat Minimal Turn   : {min_turns} giliran (min. {min_turns//2} pasang user-ai)")
    print(f"  Maksimal Karakter/Turn: {max_chars_per_turn} karakter")
    print(f"  Seed Reproduksibilitas: {seed}")
    print("=" * 76)

    total_scanned = 0
    clean_conversations = []

    with zipfile.ZipFile(zip_path, "r") as z:
        conv_files = [
            f for f in z.namelist()
            if (f.startswith("conversations-") and f.endswith(".json")) or f == "conversations.json"
        ]
        conv_files.sort()
        print(f"\n[1/3] Menemukan {len(conv_files)} berkas percakapan di dalam arsip ZIP...")

        for fname in conv_files:
            try:
                with z.open(fname) as f:
                    data = json.load(f)
                    total_scanned += len(data)
                    for conv_raw in data:
                        parsed = extract_single_conversation(
                            conv_raw,
                            min_turns=min_turns,
                            max_chars_per_turn=max_chars_per_turn
                        )
                        if parsed:
                            clean_conversations.append(parsed)
            except Exception as e:
                print(f"  ⚠️ Peringatan saat membaca {fname}: {e}")

    # Berikan ID unik berurutan
    for idx, conv in enumerate(clean_conversations):
        conv["session_index"] = idx + 1
        conv["formatted_id"] = f"chatgpt_conv_{idx+1:05d}"

    print(f"\n[2/3] Hasil Pembersihan Data:")
    print(f"  - Total sesi obrolan awal dipindai  : {total_scanned:>5,}")
    print(f"  - Sesi obrolan lolos pembersihan     : {len(clean_conversations):>5,} ({len(clean_conversations)/total_scanned*100:.1f}%)")
    total_turns = sum(c["num_turns"] for c in clean_conversations)
    print(f"  - Total giliran dialog bersih        : {total_turns:>5,} (rata-rata {total_turns/len(clean_conversations):.1f} turn/sesi)")

    # 3. Pengacakan deterministik & Pembagian Train (80%), Val (10%), Test (10%)
    random.seed(seed)
    random.shuffle(clean_conversations)

    n_train = int(len(clean_conversations) * 0.8)
    n_val = int(len(clean_conversations) * 0.1)

    train_data = clean_conversations[:n_train]
    val_data = clean_conversations[n_train:n_train + n_val]
    test_data = clean_conversations[n_train + n_val:]

    def write_jsonl(records: List[Dict[str, Any]], filename: str) -> str:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as out_f:
            for r in records:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  ✓ {filename:<30}: {len(records):>4,} sesi ({size_kb:>7.1f} KB)")
        return filepath

    print(f"\n[3/3] Menyimpan Berkas Dataset Terbagi (Split JSONL)...")
    all_f = write_jsonl(clean_conversations, "chatgpt_conversations.jsonl")
    train_f = write_jsonl(train_data, "chatgpt_train.jsonl")
    val_f = write_jsonl(val_data, "chatgpt_val.jsonl")
    test_f = write_jsonl(test_data, "chatgpt_test.jsonl")

    # Metadata Statistik
    meta = {
        "generator": "prepare_chatgpt_conversations.py",
        "source_archive": os.path.basename(zip_path),
        "seed": seed,
        "total_scanned_conversations": total_scanned,
        "clean_conversations_count": len(clean_conversations),
        "retention_rate": round(len(clean_conversations) / total_scanned, 4),
        "total_dialogue_turns": total_turns,
        "avg_turns_per_conversation": round(total_turns / len(clean_conversations), 2),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "files": {
            "all": all_f,
            "train": train_f,
            "val": val_f,
            "test": test_f,
        }
    }
    meta_path = os.path.join(output_dir, "chatgpt_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)

    print(f"\n✓ Metadata statistik disimpan di: {meta_path}")
    print("=" * 76)
    print("✓ SELURUH DATA PERCAKAPAN CHATGPT BERHASIL DIBERSIHKAN DAN SIAP DILATIH!")
    print("=" * 76)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser & Cleaner Ekspor Percakapan ChatGPT untuk Model Memory Bank")
    parser.add_argument("--zip_path", type=str, default=None, help="Path ke berkas ZIP ekspor ChatGPT")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Direktori target penyimpanan dataset")
    parser.add_argument("--min_turns", type=int, default=4, help="Minimal giliran pesan (user + assistant)")
    parser.add_argument("--max_chars_per_turn", type=int, default=3000, help="Maksimal karakter per giliran dialog")
    parser.add_argument("--seed", type=int, default=42, help="Seed deterministik untuk pembagian train/val/test")
    args = parser.parse_args()

    resolved_zip = args.zip_path if args.zip_path else find_latest_chatgpt_zip()
    if not resolved_zip:
        print("❌ Gagal menemukan berkas ZIP ChatGPT secara otomatis. Harap tentukan dengan --zip_path.")
        sys.exit(1)

    parse_chatgpt_archive(
        zip_path=resolved_zip,
        output_dir=args.output_dir,
        min_turns=args.min_turns,
        max_chars_per_turn=args.max_chars_per_turn,
        seed=args.seed,
    )
