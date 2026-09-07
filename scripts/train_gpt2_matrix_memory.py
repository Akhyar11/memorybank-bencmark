"""
scripts/train_gpt2_matrix_memory.py
===================================
Training Script for GPT-2 + Differentiable Memory Matrix (Continuous M in R^(128 x 768)).

Trainable Parameters:
  - W_q (query_encoder): R^(768 -> 768)
  - W_f (fusion_proj): R^(1536 -> 768)
Frozen:
  - GPT-2 Backbone (124M params)
  - Memory Matrix M (State buffer, requires_grad = False)

Flow per Turn:
  1. Forward pass on [Prompt ; AI] with turn-level broadcasted memory m = (1/d) * (q @ M^T) @ M.
  2. Compute NTP Cross-Entropy loss on AI response tokens only (-100 on prompt).
  3. Backward & Optimizer step updates W_q and W_f.
  4. Write prompt and AI final representations to M.
  5. Reset memory at conversation boundary (session isolation).
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.gpt2_matrix_memory_model import GPT2MatrixMemoryModel
from models.seed import set_seed


def load_conversations(data_path: str) -> Tuple[List[List[Tuple[str, str]]], str]:
    """
    Loads multi-turn conversations from a JSONL file or single-turn QA from CSV.
    Automatically resolves relative paths against PROJECT_ROOT to avoid CWD issues.
    """
    resolved_path = data_path
    if not os.path.isabs(resolved_path) and not os.path.exists(resolved_path):
        candidates = [
            os.path.join(PROJECT_ROOT, data_path),
            os.path.join(PROJECT_ROOT, "dataset", os.path.basename(data_path)),
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break

    if not os.path.exists(resolved_path):
        print(f"⚠️ PERINGATAN: File '{data_path}' tidak ditemukan di sistem!")
        print("  Menggunakan fallback dataset minimal (1 percakapan dummy).")
        return [
            [
                ("Halo, perkenalkan namaku Akhyar.", "Halo Akhyar! Senang berkenalan denganmu. Ada yang bisa dibantu?"),
                ("Aku bekerja sebagai Machine Learning Engineer.", "Keren sekali! Bidang AI dan Machine Learning sangat menjanjikan."),
                ("Siapa namaku dan apa pekerjaanku?", "Namamu adalah Akhyar dan kamu bekerja sebagai Machine Learning Engineer."),
            ]
        ], resolved_path

    conversations: List[List[Tuple[str, str]]] = []

    # Format 1: CSV QA Faktual (misalnya dataset/train.csv dengan 120.000 sampel)
    if resolved_path.endswith(".csv"):
        import csv
        with open(resolved_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fact = row.get("write_fact_A", "").strip()
                query = row.get("query_B", "").strip()
                ans = row.get("expected_output_A", "").strip()
                if fact and query and ans:
                    # 2-turn dialog: Turn 1 ingest fakta ke memory, Turn 2 recall jawaban
                    conv = [
                        (fact, "Baik, informasi tersebut telah saya catat ke dalam memori."),
                        (query, f"Jawabannya adalah {ans}."),
                    ]
                    conversations.append(conv)
        print(f"✓ Berhasil memuat {len(conversations):,} sampel dari file CSV: {resolved_path}")
        return conversations, resolved_path

    # Format 2: JSONL Multi-Turn (misalnya dataset/conversations_train.jsonl)
    with open(resolved_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            turns = item.get("turns", [])
            if isinstance(turns, list) and len(turns) >= 2:
                conv_turns: List[Tuple[str, str]] = []
                i = 0
                while i < len(turns):
                    t1 = turns[i] if isinstance(turns[i], dict) else {}
                    role1 = t1.get("role", "")
                    content1 = t1.get("content", "").strip()
                    if role1 == "user" and i + 1 < len(turns):
                        t2 = turns[i + 1] if isinstance(turns[i + 1], dict) else {}
                        role2 = t2.get("role", "")
                        content2 = t2.get("content", "").strip()
                        if role2 == "assistant" and content1 and content2:
                            conv_turns.append((content1, content2))
                            i += 2
                            continue
                    i += 1
                if conv_turns:
                    conversations.append(conv_turns)

    if not conversations:
        print(f"⚠️ PERINGATAN: Tidak ada giliran dialog valid di '{resolved_path}'. Menggunakan fallback dummy.")
        return [
            [
                ("Halo, perkenalkan namaku Akhyar.", "Halo Akhyar! Senang berkenalan denganmu. Ada yang bisa dibantu?"),
                ("Aku bekerja sebagai Machine Learning Engineer.", "Keren sekali!"),
            ]
        ], resolved_path

    print(f"✓ Berhasil memuat {len(conversations):,} percakapan dari file JSONL: {resolved_path}")
    return conversations, resolved_path


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 72)
    print("   TRAINING GPT-2 + DIFFERENTIABLE MEMORY MATRIX (CONTINUOUS M^T M q)")
    print("=" * 72)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    freeze_backbone = not args.unfreeze_backbone
    model = GPT2MatrixMemoryModel(
        model_name_or_path=args.model_dir,
        capacity=args.capacity,
        scaling=args.scaling,
        freeze_backbone=freeze_backbone,
    ).to(device)
    model.print_trainable_parameters()

    conversations, resolved_source = load_conversations(args.data_path)
    if args.max_samples > 0:
        conversations = conversations[:args.max_samples]

    total_turns = sum(len(c) for c in conversations)
    if total_turns == 0:
        raise ValueError("Tidak ada turn dialog valid untuk training.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    os.makedirs(args.output_dir, exist_ok=True)
    history = []
    best_loss = float("inf")

    print(f"Sumber Data      : {resolved_source}")
    print(f"Total percakapan : {len(conversations):,}")
    print(f"Total turn dialog: {total_turns:,}")
    print(f"Epochs           : {args.epochs}")
    print(f"Learning Rate    : {args.lr}")
    print(f"Scaling Mode     : {args.scaling}")
    print(f"Boundary Policy  : Reset M at end of each conversation (session isolation).\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(conversations, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for conv_turns in pbar:
            # Boundary percakapan: reset memory matrix
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

                # Forward pass & loss
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

                # State update: Tulis ke Memory Matrix setelah token dihitung
                with torch.no_grad():
                    # Memory 1: Prompt
                    prompt_out = model.gpt2.transformer(input_ids=prompt_ids)
                    h_prompt = prompt_out.last_hidden_state[:, -1, :]
                    model.matrix_bank.write(h_prompt)

                    # Memory 2: AI response
                    ai_out = model.gpt2.transformer(input_ids=full_ids)
                    h_ai = ai_out.last_hidden_state[:, -1, :]
                    model.matrix_bank.write(h_ai)

                total_loss += loss.item()
                steps += 1

            model.reset_memory()

            avg_loss = total_loss / max(steps, 1)
            ppl = math.exp(min(avg_loss, 20.0))
            pbar.set_postfix({
                "ntp_loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "ppl": f"{ppl:.2f}",
                "slots": f"{model.matrix_bank.num_memories}",
            })

        epoch_loss = total_loss / max(steps, 1)
        epoch_ppl = math.exp(min(epoch_loss, 20.0))
        history.append({"epoch": epoch, "ntp_loss": epoch_loss, "perplexity": epoch_ppl})
        print(f"Epoch {epoch}: ntp_loss={epoch_loss:.4f}, perplexity={epoch_ppl:.2f}")

        payload = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "capacity": args.capacity,
                "scaling": args.scaling,
            },
            "history": history,
        }

        latest_path = os.path.join(args.output_dir, "gpt2_matrix_memory_latest.pt")
        torch.save(payload, latest_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(args.output_dir, "gpt2_matrix_memory_best.pt")
            torch.save(payload, best_path)
            print(f"✓ Model terbaik tersimpan di {best_path} (Loss: {best_loss:.4f})")

    history_path = os.path.join(args.output_dir, "training_matrix_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "objective": "differentiable_matrix_memory_ntp",
            "epochs": args.epochs,
            "max_seq_len": args.max_seq_len,
            "learning_rate": args.lr,
            "best_ntp_loss": best_loss,
            "history": history,
        }, f, indent=2)

    print("\n✓ Training selesai!")


def main():
    parser = argparse.ArgumentParser(description="Training GPT-2 + Differentiable Memory Matrix")
    parser.add_argument("--data_path", type=str, default="dataset/conversations_train.jsonl")
    parser.add_argument("--model_dir", type=str, default="gpt2-indo-instruct-tuned")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--scaling", type=str, default="dim")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--unfreeze_backbone", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
