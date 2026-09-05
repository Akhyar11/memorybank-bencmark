"""
Script Training: GPT-2 + Mature TinyMemoryBank (Episodic Turn Training)
======================================================================
Repo: MemoryBank-bencmark/scripts/train_gpt2_mature_memory.py
Melatih parameter-efficient Memory Bank adapter di atas GPT-2 Indo (100% frozen)
menggunakan paradigma murni Episodic Memory:
- Langkah 1 (Write): Model memproses giliran fakta dan menyimpannya ke TinyMemoryBank.
- Langkah 2 (Recall): Prompt di-reset menjadi SINGKAT (hanya kalimat tanya, tanpa contekan teks).
- Model dipaksa 100% menggunakan representasi laten TinyMemoryBank untuk memprediksi jawaban.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import TinyMemoryConfig


class EpisodicConversationDataset(Dataset):
    """Dataset dialog percakapan untuk pelatihan episodik Memory Bank (Fact Write -> Short Recall)."""
    def __init__(self, data_path, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            self.samples.append(item)
                        except Exception:
                            pass

        if len(self.samples) == 0:
            # Fallback jika file dataset belum ada
            self.samples = [{
                "turns": [
                    {"role": "user", "content": "Halo, perkenalkan namaku Budi dan aku tinggal di Bandung."},
                    {"role": "assistant", "content": "Senang berkenalan denganmu Budi di Bandung!"}
                ],
                "facts": [{"turn": 0, "key": "name", "value": "Budi"}, {"turn": 0, "key": "city", "value": "Bandung"}],
                "target_recall": {
                    "question": "Siapa namaku dan di mana aku tinggal?",
                    "answer": "Namamu Budi dan kamu tinggal di Bandung."
                }
            }] * 50

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 1. Ekstraksi giliran fakta yang akan ditulis ke memory bank
        fact_turns = []
        if "facts" in item and len(item["facts"]) > 0:
            for f in item["facts"]:
                t_idx = f.get("turn", 0)
                if t_idx < len(item["turns"]):
                    u_t = item["turns"][t_idx]["content"]
                    ai_t = item["turns"][t_idx + 1]["content"] if t_idx + 1 < len(item["turns"]) else "Baik."
                    fact_turns.append((u_t, ai_t))
        if not fact_turns:
            if "turns" in item and len(item["turns"]) >= 2:
                fact_turns.append((item["turns"][0]["content"], item["turns"][1]["content"]))
            else:
                fact_turns.append(("Namaku Akhyar.", "Halo Akhyar."))

        # 2. Ekstraksi pertanyaan recall singkat dan target jawaban
        if "target_recall" in item and isinstance(item["target_recall"], dict):
            recall_q = item["target_recall"].get("question", "Siapa namaku?")
            recall_a = item["target_recall"].get("answer", "Namamu tercatat di memori.")
        elif "turns" in item and len(item["turns"]) >= 2:
            recall_q = item["turns"][-2]["content"]
            recall_a = item["turns"][-1]["content"]
        else:
            recall_q = "Siapa namaku?"
            recall_a = "Namamu Akhyar."

        return {
            "fact_turns": fact_turns,  # Seluruh fakta yang ada pada episode percakapan
            "recall_query": recall_q,
            "recall_answer": recall_a
        }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Menggunakan device: {device}")

    model_dir = "/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/gpt2-indo-instruct-tuned"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mem_config = TinyMemoryConfig(
        memory_capacity=args.capacity,
        memory_dim=768,
        hidden_size=768,
        memory_read_threshold=0.2  # Ambang batas responsif untuk fusi saraf
    )

    print("Memuat GPT2MemoryModel dengan mature TinyMemoryBank...")
    model = GPT2MemoryModel(
        model_name_or_path=model_dir,
        memory_config=mem_config,
        freeze_backbone=True
    ).to(device)

    model.print_trainable_parameters()

    dataset = EpisodicConversationDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    grad_accum_steps = max(1, args.batch_size)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    os.makedirs(args.output_dir, exist_ok=True)

    def run_episodic_step(item):
        """Menjalankan satu siklus lengkap: Write fakta -> Recall prompt singkat."""
        model.reset_memory()

        # Langkah 1: Write Fakta ke TinyMemoryBank (Konsep N-Token Representasi)
        write_loss_accum = torch.tensor(0.0, device=device)
        for u_text, ai_text in item["fact_turns"]:
            write_text = f"User: {u_text}\nAI: {ai_text}"
            w_enc = tokenizer(write_text, max_length=args.max_seq_len, truncation=True, return_tensors="pt").to(device)
            
            pfx_ids = tokenizer("User: ", return_tensors="pt")["input_ids"]
            pfx_len = pfx_ids.shape[1]
            u_enc = tokenizer(f"User: {u_text}\n", return_tensors="pt")
            u_len = min(u_enc["input_ids"].shape[1], w_enc["input_ids"].shape[1])

            write_targets = torch.zeros_like(w_enc["input_ids"]).float().to(device)
            if u_len > pfx_len + 1:
                write_targets[0, pfx_len:u_len-1] = 1.0

            out_w = model(
                input_ids=w_enc["input_ids"],
                attention_mask=w_enc["attention_mask"],
                write_targets=write_targets
            )
            if out_w["loss"] is not None:
                write_loss_accum = write_loss_accum + out_w["loss"]

            # N-Token Representasi (N=8, Opsi 1: Attention-Weighted Soft Pooling)
            user_hiddens = out_w["hidden_states"][0, pfx_len:u_len-1]
            user_w_probs = out_w["write_probs"][0, pfx_len:u_len-1]
            if user_hiddens.size(0) > 0:
                model.write_memory_chunked(user_hiddens, user_w_probs, chunk_size=8)

        # Langkah 2: Recall dengan Prompt Singkat (Tanpa teks fakta di prompt!)
        query_prompt = f"User: {item['recall_query']}\nAI:"
        target_answer = f" {item['recall_answer']}"
        full_recall = query_prompt + target_answer

        q_ids = tokenizer(query_prompt, return_tensors="pt")["input_ids"]
        q_len = q_ids.shape[1]

        full_enc = tokenizer(full_recall, max_length=args.max_seq_len, truncation=True, return_tensors="pt").to(device)
        recall_ids = full_enc["input_ids"]
        recall_mask = full_enc["attention_mask"]

        recall_labels = recall_ids.clone()
        recall_labels[0, :q_len] = -100  # Zero loss pada pertanyaan; model hanya dinilai pada target jawaban

        read_targets = torch.zeros_like(recall_ids).float().to(device)
        read_targets[0, :q_len] = 1.0   # Supervisi read_head agar membuka gerbang memori saat mendeteksi pertanyaan

        out_r = model(
            input_ids=recall_ids,
            attention_mask=recall_mask,
            labels=recall_labels,
            read_targets=read_targets
        )

        step_loss = out_r["loss"] + 0.3 * write_loss_accum
        return step_loss, out_r["loss"].item()

    if args.dry_run:
        print("\n[DRY RUN]: Menguji 1 siklus episodik (Write -> Short Recall)...")
        item = dataset[0]
        step_loss, recall_loss = run_episodic_step(item)
        print(f"[DRY RUN SUKSES]: Initial Recall Loss = {recall_loss:.4f}")
        return

    print(f"\nMemulai Pelatihan Episodik TinyMemoryBank ({args.epochs} Epochs, Grad Accum: {grad_accum_steps})...")
    model.train()
    history = []
    best_loss = float("inf")

    def save_adapter_checkpoint(filename: str, extra_meta: dict = None):
        """Helper untuk menyimpan seluruh parameter adapter memory bank secara aman."""
        save_path = os.path.join(args.output_dir, filename)
        adapter_state = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("gpt2.")}
        payload = {
            "adapter_state_dict": adapter_state,
            "bank": model.bank.state_dict(),
            "write_head": model.write_head.state_dict(),
            "read_head": model.read_head.state_dict(),
            "config": mem_config,
            "history": history,
            "extra_meta": extra_meta or {}
        }
        torch.save(payload, save_path)
        return save_path

    try:
        for epoch in range(1, args.epochs + 1):
            total_loss = 0.0
            num_samples = 0
            optimizer.zero_grad()

            pbar = tqdm(
                range(len(dataset)),
                desc=f"Epoch [{epoch}/{args.epochs}]",
                unit="dialog",
                dynamic_ncols=True,
                leave=True
            )
            for step in pbar:
                item = dataset[step]
                step_loss, recall_loss_val = run_episodic_step(item)

                # Akumulasi gradien
                loss_scaled = step_loss / grad_accum_steps
                loss_scaled.backward()

                total_loss += recall_loss_val
                num_samples += 1
                running_avg = total_loss / num_samples

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataset):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                pbar.set_postfix({
                    "recall_loss": f"{recall_loss_val:.4f}",
                    "avg_loss": f"{running_avg:.4f}",
                    "slots": model.bank.active_count
                })

            avg_loss = total_loss / max(1, num_samples)
            history.append({"epoch": epoch, "loss": avg_loss})
            print(f"\n✓ Epoch [{epoch}/{args.epochs}] Selesai - Rata-rata Recall Loss: {avg_loss:.4f}")

            # Simpan best model jika loss membaik
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_adapter_checkpoint("best_adapter.pt", {"best_epoch": epoch, "best_loss": best_loss})
                print(f"  └─ ★ Best loss baru ({best_loss:.4f})! Disimpan ke best_adapter.pt")

    except KeyboardInterrupt:
        print("\n[PERINGATAN]: Pelatihan dihentikan oleh user (Ctrl+C). Menyimpan checkpoint terakhir...")
        interrupted_path = save_adapter_checkpoint("gpt2_mature_memory_adapter.pt", {"status": "interrupted", "history": history})
        print(f"✓ Checkpoint darurat berhasil disimpan ke: {interrupted_path}")
        return

    # Simpan checkpoint final standar
    final_path = save_adapter_checkpoint("gpt2_mature_memory_adapter.pt", {"best_loss": best_loss, "final_loss": avg_loss})

    # Simpan riwayat training ke JSON untuk audit
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "GPT-2 Indo + Mature TinyMemoryBank (Episodic)",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "best_loss": best_loss,
            "history": history
        }, f, indent=2)

    print(f"\n=======================================================")
    print(f"✓ SELURUH CHECKPOINT BERHASIL DISIMPAN:")
    print(f"  1. Adapter Final : {final_path}")
    print(f"  2. Adapter Best  : {os.path.join(args.output_dir, 'best_adapter.pt')}")
    print(f"  3. History Log   : {history_path}")
    print(f"=======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Episodic Training: Mature TinyMemoryBank on GPT-2 Indo")
    parser.add_argument("--data_path", type=str, default="dataset/conversations_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4, help="Gradient accumulation batch size")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Uji 1 siklus tanpa training")

    args = parser.parse_args()
    train(args)
