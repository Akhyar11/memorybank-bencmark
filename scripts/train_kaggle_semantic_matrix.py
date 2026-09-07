"""
scripts/train_kaggle_semantic_matrix.py
========================================
Skrip Pelatihan Kaggle GPU (T4 / P100) untuk Pendekatan 1:
Melatih Adapter Fusi Memori dengan Semantic Sentence Extractor (IndoBERT).

Cara Menjalankan di Kaggle:
  python train_kaggle_semantic_matrix.py \
      --dataset dataset/conversations_train.jsonl \
      --epochs 3 \
      --lr 2e-4 \
      --output_dir checkpoints
"""

import os
import sys
import math
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.gpt2_matrix_memory_model import GPT2MatrixMemoryModel
from models.semantic_extractor import SemanticSentenceExtractor


def load_conversations(file_path: str):
    conversations = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            turns = item.get("turns", [])
            if not turns:
                continue

            dialog_pairs = []
            u_text = None
            for t in turns:
                role = t.get("role", "")
                content = t.get("content", "").strip()
                if role == "user":
                    u_text = content
                elif role == "assistant" and u_text is not None:
                    dialog_pairs.append((u_text, content))
                    u_text = None

            if dialog_pairs:
                conversations.append(dialog_pairs)
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Train Semantic Matrix Memory on Kaggle")
    parser.add_argument("--model_name", type=str, default="izzulgod/gpt2-indo-instruct-tuned")
    parser.add_argument("--bert_name", type=str, default="indolem/indobert-base-uncased")
    parser.add_argument("--dataset", type=str, default="dataset/conversations_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[1/4] Memuat SemanticSentenceExtractor...")
    extractor = SemanticSentenceExtractor(
        extractor_type="indobert",
        model_name_or_path=args.bert_name,
        device=device,
    )

    print("\n[2/4] Memuat GPT2MatrixMemoryModel...")
    model = GPT2MatrixMemoryModel(
        model_name_or_path=args.model_name,
        capacity=128,
        scaling="dim",
        freeze_backbone=True,
        semantic_extractor=extractor,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    print("\n[3/4] Memuat Data Percakapan...")
    conversations = load_conversations(args.dataset)
    print(f"Total percakapan: {len(conversations):,}")

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")

    print("\n[4/4] Memulai Pelatihan...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(conversations, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for conv_turns in pbar:
            model.reset_memory()

            for user_text, ai_text in conv_turns:
                prompt_str = f"User: {user_text}\nAI:"
                ai_str = f" {ai_text}\n"

                prompt_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(device)
                ai_ids = tokenizer(ai_str, return_tensors="pt")["input_ids"].to(device)
                full_ids = torch.cat([prompt_ids, ai_ids], dim=-1)

                if full_ids.size(1) > args.max_seq_len:
                    full_ids = full_ids[:, :args.max_seq_len]
                    prompt_len = min(prompt_ids.size(1), args.max_seq_len - 1)
                else:
                    prompt_len = prompt_ids.size(1)

                if full_ids.size(1) <= prompt_len:
                    continue

                labels = full_ids.clone()
                labels[:, :prompt_len] = -100

                out = model(
                    input_ids=full_ids,
                    labels=labels,
                    use_memory=True,
                    prompt_len=prompt_len,
                )

                loss = out["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                # Tulis representasi semantik kalimat ke Matrix Bank
                with torch.no_grad():
                    v_user = extractor.encode(user_text, normalize=True)
                    model.matrix_bank.write(v_user)

                total_loss += loss.item()
                steps += 1

            model.reset_memory()

        avg_loss = total_loss / max(steps, 1)
        ppl = math.exp(min(avg_loss, 20.0))
        print(f"\nEpoch {epoch} Selesai. Avg Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_file = os.path.join(args.output_dir, "gpt2_semantic_matrix_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_loss": best_loss,
            }, ckpt_file)
            print(f"✓ Checkpoint terbaik disimpan ke: {ckpt_file}")

    print("\nPelatihan selesai dengan sukses!")


if __name__ == "__main__":
    main()
