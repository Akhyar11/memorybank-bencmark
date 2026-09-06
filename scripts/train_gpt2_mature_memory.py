"""
scripts/train_gpt2_mature_memory.py
===================================
Training Turn-Level Semantic Memory Bank for GPT-2 (NTP-only).

Trainable Parameters:
  - W_c (c_proj): R^(768 -> 768)
  - W_f (fusion_proj): R^(1536 -> 768)
Backbone GPT-2 and LM Head: 100% FROZEN.

Flow per Conversation Turn:
  1. User Prompt enters -> Forward -> Save Memory 1 (h_prompt).
  2. Read Memory using C = h_prompt * W_c.
  3. Predict Assistant Response tokens with fusion z = [h ; M_bar] * W_f -> LM Head.
  4. Compute NTP loss on assistant response tokens, backward & optimizer step.
  5. Save Memory 2 (h_ai_final).
  6. Lifecycle turn step & eviction.
  7. End of conversation: reset memory for clean session boundary.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import TinyMemoryConfig


def load_conversations(data_path: str) -> List[List[Tuple[str, str]]]:
    """
    Loads multi-turn conversations from a JSONL file.
    Returns a list of conversations, where each conversation is a list of (user_msg, assistant_msg) turns.
    """
    conversations: List[List[Tuple[str, str]]] = []
    if not os.path.exists(data_path):
        return [
            [
                ("Halo, perkenalkan namaku Akhyar.", "Halo Akhyar! Senang berkenalan denganmu. Ada yang bisa dibantu?"),
                ("Aku bekerja sebagai Programmer Python.", "Keren sekali! Programmer Python sangat banyak diminati saat ini."),
                ("Siapa namaku dan apa pekerjaanku?", "Namamu adalah Akhyar dan kamu bekerja sebagai Programmer Python."),
            ]
        ]

    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
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

    return conversations if conversations else [
        [
            ("Halo, perkenalkan namaku Akhyar.", "Halo Akhyar! Senang berkenalan denganmu. Ada yang bisa dibantu?"),
            ("Aku bekerja sebagai Programmer Python.", "Keren sekali! Programmer Python sangat banyak diminati saat ini."),
        ]
    ]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 68)
    print("   TRAINING GPT-2 + TURN-LEVEL SEMANTIC MEMORY BANK (NTP-ONLY)")
    print("=" * 68)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mem_config = TinyMemoryConfig(
        memory_capacity=args.capacity,
        memory_dim=768,
        hidden_size=768,
        top_k=args.top_k,
    )

    freeze_backbone = not args.unfreeze_backbone
    model = GPT2MemoryModel(
        model_name_or_path=args.model_dir,
        memory_config=mem_config,
        freeze_backbone=freeze_backbone,
    ).to(device)
    model.print_trainable_parameters()

    conversations = load_conversations(args.data_path)
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

    print(f"Total percakapan : {len(conversations)}")
    print(f"Total turn dialog: {total_turns}")
    print(f"Memulai training turn-level selama {args.epochs} epoch (backbone frozen: {freeze_backbone})...")
    print(f"Boundary policy  : Memori di-reset di setiap akhir percakapan (clean session isolation).\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(conversations, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for conv_turns in pbar:
            # Boundary percakapan: reset memori untuk sesi baru
            model.reset_memory()

            for user_text, ai_text in conv_turns:
                # 1. Format User Prompt dan Assistant Response
                prompt_str = f"User: {user_text}\nAI:"
                ai_str = f" {ai_text}\n"

                prompt_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(device)
                ai_ids = tokenizer(ai_str, return_tensors="pt")["input_ids"].to(device)

                # Batasi panjang sekuens jika melebihi max_seq_len
                full_ids = torch.cat([prompt_ids, ai_ids], dim=-1)
                if full_ids.size(1) > args.max_seq_len:
                    full_ids = full_ids[:, :args.max_seq_len]
                    prompt_len = min(prompt_ids.size(1), args.max_seq_len - 1)
                else:
                    prompt_len = prompt_ids.size(1)

                if full_ids.size(1) <= prompt_len:
                    continue

                # Target labels: hanya hitung loss pada token respons AI (-100 pada prompt)
                labels = full_ids.clone()
                labels[:, :prompt_len] = -100

                # 2. Forward pass & compute NTP loss dengan Memory Retrieval
                out = model(
                    input_ids=full_ids,
                    labels=labels,
                    use_memory=True,
                    persist_memory=False,
                )

                loss = out["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                # 3. Update Memory Bank untuk Turn ini (2 memori):
                with torch.no_grad():
                    # Memory 1: Token terakhir prompt
                    prompt_out = model.gpt2.transformer(input_ids=prompt_ids)
                    h_prompt = prompt_out.last_hidden_state[:, -1, :]
                    model.bank.add_memory(h_prompt.detach())

                    # Memory 2: Token terakhir respon AI
                    ai_out = model.gpt2.transformer(input_ids=full_ids)
                    h_ai = ai_out.last_hidden_state[:, -1, :]
                    model.bank.add_memory(h_ai.detach())

                # Step turn & eviction lifecycle
                model.bank.step_turn()
                model.bank.evict_lifecycle()

                total_loss += loss.item()
                steps += 1

            # Akhir percakapan: reset memori
            model.reset_memory()

            avg_loss = total_loss / max(steps, 1)
            ppl = math.exp(min(avg_loss, 20.0))
            diag = model.last_diagnostics
            pbar.set_postfix({
                "ntp_loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "ppl": f"{ppl:.2f}",
                "mems": f"{diag.get('num_memories', 0.0):.0f}",
                "reads": f"{diag.get('mean_reads', 0.0):.1f}",
            })

        epoch_loss = total_loss / max(steps, 1)
        epoch_ppl = math.exp(min(epoch_loss, 20.0))
        history.append({"epoch": epoch, "ntp_loss": epoch_loss, "perplexity": epoch_ppl})
        print(f"Epoch {epoch}: ntp_loss={epoch_loss:.4f}, perplexity={epoch_ppl:.2f}")

        payload = {
            "model_state_dict": model.state_dict(),
            "memory_config": vars(mem_config),
            "history": history,
        }

        latest_path = os.path.join(args.output_dir, "gpt2_causal_memory_latest.pt")
        torch.save(payload, latest_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(args.output_dir, "gpt2_causal_memory_best.pt")
            torch.save(payload, best_path)

    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "objective": "turn_level_semantic_memory_ntp",
                "epochs": args.epochs,
                "max_seq_len": args.max_seq_len,
                "learning_rate": args.lr,
                "best_ntp_loss": best_loss,
                "history": history,
            },
            f,
            indent=2,
        )

    print("✓ Selesai. Checkpoint dan riwayat training sudah disimpan.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training GPT-2 + Turn-Level Semantic Memory Bank")
    parser.add_argument("--data_path", type=str, default="dataset/conversations_train.jsonl")
    parser.add_argument("--model_dir", type=str, default="gpt2-indo-instruct-tuned")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=-1, help="Batas jumlah percakapan untuk training (-1 untuk semua data)")
    parser.add_argument("--top_k", type=int, default=1, help="Top-K memories to retrieve")
    parser.add_argument("--unfreeze_backbone", action="store_true", default=False, help="Unfreeze GPT-2 backbone (default is frozen)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    train(args)
