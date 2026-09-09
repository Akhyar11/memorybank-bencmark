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
from models.seed import set_seed


def load_conversations(file_path: str, seed: int = 42):
    resolved_path = file_path
    if not os.path.isabs(resolved_path) and not os.path.exists(resolved_path):
        candidates = [
            os.path.join(PROJECT_ROOT, file_path),
            os.path.join(PROJECT_ROOT, "dataset", os.path.basename(file_path)),
            "/kaggle/input/datasets/akhyarsafrudin/memorybank-benchmark/8766716822ea6bbc15133b9fdd644dee89186edc0d9502c9bc5d288d4efae226-2026-09-05-16-08-32-4dc7aafcc52a4a7482a7e83d419d581d/conversations_train.jsonl",
            "/kaggle/input/memorybank-benchmark/8766716822ea6bbc15133b9fdd644dee89186edc0d9502c9bc5d288d4efae226-2026-09-05-16-08-32-4dc7aafcc52a4a7482a7e83d419d581d/conversations_train.jsonl",
            "/kaggle/input/datasets/akhyarsafrudin/memorybank-benchmark/conversations_train.jsonl",
            "/kaggle/input/memorybank-benchmark/conversations_train.jsonl",
            "/kaggle/input/conversations_train.jsonl",
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break
        if not os.path.exists(resolved_path) and os.path.isdir("/kaggle/input"):
            found = glob.glob(f"/kaggle/input/**/{os.path.basename(file_path)}", recursive=True)
            if found:
                resolved_path = found[0]

    if not os.path.exists(resolved_path):
        print(f"ℹ File '{file_path}' tidak ditemukan. Mengenerate dataset otomatis (1,000 percakapan)...")
        from scripts.generate_conversation_dataset import generate_conversation_dataset
        ds_meta = generate_conversation_dataset(
            num_conversations=1000,
            seed=seed,
            output_dir=os.path.join(PROJECT_ROOT, "dataset"),
        )
        resolved_path = ds_meta.get("train_file", os.path.join(PROJECT_ROOT, "dataset", "conversations_train.jsonl"))

    conversations = []
    with open(resolved_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            turns = item.get("turns", [])
            if not turns:
                continue

            target_recall = item.get("target_recall", {})
            target_question = target_recall.get("question", "").strip() if target_recall else ""

            dialog_pairs = []
            u_text = None
            for t in turns:
                role = t.get("role", "")
                content = t.get("content", "").strip()
                if role == "user":
                    u_text = content
                elif role == "assistant" and u_text is not None:
                    is_recall = False
                    if target_question and (u_text == target_question or target_question in u_text or u_text in target_question):
                        is_recall = True
                    dialog_pairs.append((u_text, content, is_recall))
                    u_text = None

            if dialog_pairs:
                if target_recall and not any(p[2] for p in dialog_pairs):
                    u_last, a_last, _ = dialog_pairs[-1]
                    dialog_pairs[-1] = (u_last, a_last, True)
                conversations.append(dialog_pairs)

    print(f"✓ Sumber Data Terpakai: {resolved_path}")
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
    parser.add_argument("--recall_loss_weight", type=float, default=4.0, help="Loss multiplier for target recall turns (default: 4.0)")
    parser.add_argument("--reset_memory_per_conv", action="store_true", default=False, help="Reset memory at the beginning of each conversation (default: False, continuous lifelong rolling memory)")
    parser.add_argument("--scaling", type=str, default="none", choices=["none", "sqrt", "dim"], help="Scaling factor for memory attention: none (1.0), sqrt (1/sqrt(d)), dim (1/d)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Random Seed: {args.seed}")
    print(f"Device     : {device}")
    print(f"Scaling    : {args.scaling}")

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
        scaling=args.scaling,
        freeze_backbone=True,
        semantic_extractor=extractor,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    print("\n[3/4] Memuat Data Percakapan...")
    conversations = load_conversations(args.dataset, seed=args.seed)
    print(f"Total percakapan: {len(conversations):,}")

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")

    print("\n[4/4] Memulai Pelatihan...")
    model.reset_memory()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recall_loss = 0.0
        steps = 0
        recall_steps = 0

        pbar = tqdm(conversations, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for conv_turns in pbar:
            if args.reset_memory_per_conv:
                model.reset_memory()

            for user_text, ai_text, is_recall in conv_turns:
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
                    query_text=user_text if model.semantic_extractor is not None else None,
                )

                loss = out["loss"]
                raw_loss = loss.item()

                if is_recall and args.recall_loss_weight > 1.0:
                    train_loss = loss * args.recall_loss_weight
                else:
                    train_loss = loss

                optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                # Tulis representasi semantik kalimat ke Matrix Bank (KEDUA bagian: User dan AI)
                with torch.no_grad():
                    v_user = extractor.encode(f"User: {user_text}", normalize=True)
                    v_ai = extractor.encode(f"AI: {ai_text}", normalize=True)
                    model.matrix_bank.write(v_user)
                    model.matrix_bank.write(v_ai)

                total_loss += raw_loss
                steps += 1
                if is_recall:
                    total_recall_loss += raw_loss
                    recall_steps += 1

            if args.reset_memory_per_conv:
                model.reset_memory()

            avg_l = total_loss / max(steps, 1)
            avg_rec = total_recall_loss / max(recall_steps, 1) if recall_steps > 0 else 0.0
            pbar.set_postfix({
                "Loss": f"{avg_l:.4f}",
                "RecLoss": f"{avg_rec:.4f}",
                "Slots": model.matrix_bank.num_memories,
            })

        avg_loss = total_loss / max(steps, 1)
        avg_rec_loss = total_recall_loss / max(recall_steps, 1) if recall_steps > 0 else 0.0
        ppl = math.exp(min(avg_loss, 20.0))
        print(f"\nEpoch {epoch} Selesai. Avg Loss: {avg_loss:.4f} (Recall: {avg_rec_loss:.4f}) | Perplexity: {ppl:.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Simpan HANYA bobot adapter trainable (query_encoder & fusion_proj) -> Ukuran HANYA ~7 MB!
            adapter_sd = model.get_adapter_state_dict()
            ckpt_file = os.path.join(args.output_dir, "matrix_adapter_best.pt")
            torch.save({
                "epoch": epoch,
                "adapter_state_dict": adapter_sd,
                "best_loss": best_loss,
                "config": {
                    "capacity": 128,
                    "scaling": args.scaling,
                    "model_name": args.model_name,
                    "bert_name": args.bert_name,
                    "recall_loss_weight": args.recall_loss_weight,
                    "reset_memory_per_conv": args.reset_memory_per_conv,
                },
            }, ckpt_file)
            print(f"✓ Checkpoint Adapter Ringan (~7 MB) disimpan ke: {ckpt_file}")

    print("\nPelatihan selesai dengan sukses!")


if __name__ == "__main__":
    main()
