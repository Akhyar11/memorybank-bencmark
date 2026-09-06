"""
scripts/test_chat_matrix_memory.py
==================================
Kode uji inferensi multi-turn menggunakan arsitektur GPT-2 + Differentiable Memory Matrix.
"""

import os
import sys
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.gpt2_matrix_memory_model import GPT2MatrixMemoryModel

# 1. Setup Device & Model Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "gpt2-indo-instruct-tuned"
if not os.path.exists(os.path.join(PROJECT_ROOT, model_dir)):
    model_dir = "izzulgod/gpt2-indo-instruct-tuned"

print(f"Menggunakan device : {device}")
print(f"Memuat tokenizer   : {model_dir}")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Inisialisasi Model Differentiable Memory Matrix (128 Slot)
model = GPT2MatrixMemoryModel(
    model_name_or_path=model_dir,
    capacity=128,
    scaling="dim",
    freeze_backbone=True,
).to(device)

# 3. Muat Checkpoint Jika Ada
ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "gpt2_matrix_memory_best.pt")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print(f"✓ Checkpoint berhasil dimuat dari: {ckpt_path}")
else:
    print(f"ℹ Checkpoint {ckpt_path} belum ditemukan, menggunakan inisialisasi bobot default.")

model.eval()
model.reset_memory()

# -------------------------------------------------------------
# Turn 1: Memberikan Fakta Pertama
# -------------------------------------------------------------
prompt1 = "User: Halo, namaku Akhyar dan aku seorang Programmer Python di Bandung.\nAI:"
inputs1 = tokenizer(prompt1, return_tensors="pt").to(device)

out1 = model.generate(inputs1["input_ids"], max_new_tokens=40, temperature=0.3)
print("\n" + "=" * 76)
print("[Turn 1]")
print(tokenizer.decode(out1[0], skip_special_tokens=True))
print(f"-> Total slot memori terisi: {model.matrix_bank.num_memories} / 128")

# -------------------------------------------------------------
# Turn 2: Menguji Penarikan (Recall) Fakta
# -------------------------------------------------------------
prompt2 = "User: Kamu masih ingat siapa namaku, apa pekerjaanku, dan di kota mana aku tinggal?\nAI:"
inputs2 = tokenizer(prompt2, return_tensors="pt").to(device)

out2 = model.generate(inputs2["input_ids"], max_new_tokens=40, temperature=0.3)
print("\n" + "=" * 76)
print("[Turn 2 - Recall]")
print(tokenizer.decode(out2[0], skip_special_tokens=True))
print(f"-> Total slot memori terisi: {model.matrix_bank.num_memories} / 128")
print("=" * 76)
