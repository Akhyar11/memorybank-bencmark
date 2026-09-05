import argparse
import json
import math
import os
import sys
from typing import List

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import TinyMemoryConfig


def _extract_text_from_item(item: dict) -> str:
    if isinstance(item, dict):
        if isinstance(item.get("text"), str) and item["text"].strip():
            return item["text"].strip()
        turns = item.get("turns")
        if isinstance(turns, list) and turns:
            pieces = []
            for t in turns:
                role = str(t.get("role", "")).strip() if isinstance(t, dict) else ""
                content = str(t.get("content", "")).strip() if isinstance(t, dict) else ""
                if content:
                    pieces.append(f"{role}: {content}" if role else content)
            if pieces:
                return "\n".join(pieces)
    return ""


def load_dialogues(data_path: str) -> List[str]:
    if not os.path.exists(data_path):
        return ["Ini adalah korpus contoh bahasa Indonesia untuk pelatihan next token prediction."]

    if data_path.endswith(".jsonl"):
        docs: List[str] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    docs.append(line)
                    continue
                text = _extract_text_from_item(item)
                if text:
                    docs.append(text)
        return docs if docs else ["Korpus kosong."]

    with open(data_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return ["Korpus kosong."]
    # Split paragraphs for plain text files
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return paragraphs if paragraphs else [raw]


def build_chunks(token_ids: torch.Tensor, seq_len: int, stride: int) -> List[torch.Tensor]:
    chunks: List[torch.Tensor] = []
    total = token_ids.numel()
    if total < 2:
        return chunks

    pos = 0
    while pos + 1 < total:
        end = min(pos + seq_len + 1, total)
        chunk = token_ids[pos:end]
        if chunk.numel() >= 2:
            chunks.append(chunk)
        if end == total:
            break
        pos += stride
    return chunks


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Menggunakan device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mem_config = TinyMemoryConfig(
        memory_capacity=args.capacity,
        memory_dim=768,
        hidden_size=768,
        tau_read=args.tau_read,
        tau_write=args.tau_write,
        lambda_replace=args.lambda_replace,
    )

    freeze_backbone = not args.unfreeze_backbone
    model = GPT2MemoryModel(
        model_name_or_path=args.model_dir,
        memory_config=mem_config,
        freeze_backbone=freeze_backbone,
    ).to(device)
    model.print_trainable_parameters()

    dialogues = load_dialogues(args.data_path)
    dialogue_chunks: List[List[torch.Tensor]] = []
    for diag_text in dialogues:
        t_ids = tokenizer(diag_text, return_tensors="pt", truncation=False)["input_ids"][0]
        cks = build_chunks(t_ids, seq_len=args.max_seq_len, stride=args.stride)
        if cks:
            dialogue_chunks.append(cks)

    total_chunks = sum(len(c) for c in dialogue_chunks)
    if total_chunks == 0:
        raise ValueError("Tidak ada chunk valid untuk training (minimal 2 token per chunk).")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    os.makedirs(args.output_dir, exist_ok=True)
    history = []
    best_loss = float("inf")

    print(f"Total percakapan : {len(dialogue_chunks)}")
    print(f"Total chunk token: {total_chunks}")
    print(f"Memulai training NTP-only selama {args.epochs} epoch (backbone frozen: {freeze_backbone})...")
    print(f"Boundary policy  : Memori & Graph di-reset di setiap akhir percakapan (clean isolation).")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(dialogue_chunks, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for chunks in pbar:
            # === BOUNDARY PERCAKAPAN: RESET MEMORI UNTUK SESI BARU ===
            model.reset_memory()
            memory_state = None

            for chunk in chunks:
                chunk = chunk.to(device)
                input_ids = chunk.unsqueeze(0)
                labels = input_ids.clone()

                # Forward pass through causal memory with standard NTP loss
                out = model(
                    input_ids=input_ids,
                    labels=labels,
                    memory_state=memory_state,
                    use_memory=True,
                    persist_memory=False,
                )

                loss = out["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                # Truncated-BPTT boundary between sequential chunks in the SAME conversation
                memory_state = model.detach_memory_state(out["memory_state"])

                total_loss += loss.item()
                steps += 1

            # === AKHIR PERCAKAPAN: RESET MEMORI & PUTUS GRAFIK ===
            memory_state = None
            model.reset_memory()

            avg_loss = total_loss / max(steps, 1)
            ppl = math.exp(min(avg_loss, 20.0))
            diag = model.last_diagnostics
            pbar.set_postfix({
                "ntp_loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "ppl": f"{ppl:.2f}",
                "occ_sum": f"{diag.get('occupancy_sum', 0.0):.1f}",
                "w_gate": f"{diag.get('avg_write_gate', 0.0):.3f}",
                "usage": f"{diag.get('usage_mean', 0.0):.3f}",
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
                "objective": "next_token_prediction_only",
                "epochs": args.epochs,
                "max_seq_len": args.max_seq_len,
                "stride": args.stride,
                "learning_rate": args.lr,
                "best_ntp_loss": best_loss,
                "history": history,
            },
            f,
            indent=2,
        )

    print("Selesai. Checkpoint dan riwayat training sudah disimpan.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training GPT-2 + Differentiable Causal Memory (NTP-only)")
    parser.add_argument("--data_path", type=str, default="dataset/conversations_train.jsonl")
    parser.add_argument("--model_dir", type=str, default="gpt2-indo-instruct-tuned")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tau_read", "--read_temperature", dest="tau_read", type=float, default=0.05, help="Read temperature tau_read")
    parser.add_argument("--tau_write", "--write_temperature", dest="tau_write", type=float, default=0.05, help="Write temperature tau_write")
    parser.add_argument("--lambda_replace", type=float, default=1.0, help="Replacement score weight lambda_replace")
    parser.add_argument("--unfreeze_backbone", action="store_true", default=False, help="Unfreeze GPT-2 backbone (default is frozen)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    train(args)
