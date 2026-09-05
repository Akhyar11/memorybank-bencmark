"""
Interactive Chat: GPT-2 Indo + Mature TinyMemoryBank
====================================================
Repo: MemoryBank-bencmark/scripts/chat_gpt2_mature_memory.py
Chat langsung dengan model GPT-2 Indo menggunakan modul TinyMemoryBank yang sudah teraudit.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import torch
from transformers import AutoTokenizer
from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import TinyMemoryConfig, STATE_ACTIVE, STATE_EXPIRED, STATE_DORMANT

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*68)
    print("   GPT-2 INDO + MATURE TINYMEMORYBANK INTERACTIVE CHAT (BENCHMARK)")
    print("="*68)
    print(f"Device        : {device}")

    model_dir = "/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/gpt2-indo-instruct-tuned"
    print(f"Memuat Model  : {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    mem_config = TinyMemoryConfig(
        memory_capacity=128,
        memory_dim=768,
        hidden_size=768,
        memory_read_threshold=0.2
    )
    model = GPT2MemoryModel(
        model_name_or_path=model_dir,
        memory_config=mem_config,
        freeze_backbone=True
    ).to(device)

    # Muat checkpoint adapter jika tersedia (prioritaskan best_adapter.pt jika ada)
    candidates = [
        "/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/checkpoints/best_adapter.pt",
        "/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/checkpoints/gpt2_mature_memory_adapter.pt",
    ]
    loaded = False
    for p in candidates:
        if os.path.exists(p):
            print(f"Memuat Adapter: {p}")
            try:
                ckpt = torch.load(p, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(p, map_location=device)
            if "adapter_state_dict" in ckpt:
                model.load_state_dict(ckpt["adapter_state_dict"], strict=False)
            else:
                model.bank.load_state_dict(ckpt["bank"])
                model.write_head.load_state_dict(ckpt["write_head"])
                model.read_head.load_state_dict(ckpt["read_head"])
            loaded = True
            break

    if not loaded:
        print("Menggunakan inisialisasi awal TinyMemoryBank (Zero-shot / Belum dilatih)")

    model.eval()

    print("-" * 68)
    print("✓ Model siap diajak mengobrol! Perintah khusus:")
    print("  /slots  : Melihat slot TinyMemoryBank yang sedang aktif")
    print("  /decay  : Memicu peluruhan memori (decay)")
    print("  /reset  : Mengosongkan seluruh memori bank")
    print("  /exit   : Keluar dari obrolan")
    print("-" * 68 + "\n")

    history = ""
    turn = 1

    while True:
        try:
            user_input = input(f"[Turn {turn}] Anda: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSelesai.")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("Sampai jumpa!")
            break

        if user_input == "/reset":
            model.reset_memory()
            history = ""
            turn = 1
            print("[SISTEM]: Seluruh slot TinyMemoryBank dan riwayat obrolan telah dikosongkan.\n")
            continue

        if user_input == "/decay":
            model.bank.global_step += 5
            model.bank.decay_memory()
            print("[SISTEM]: Peluruhan memori (decay_memory) telah diaplikasikan.\n")
            continue

        if user_input == "/slots":
            active_slots = (model.bank.mem_state == STATE_ACTIVE).nonzero(as_tuple=True)[0]
            print(f"\n[STATUS TINYMEMORYBANK]: {len(active_slots)}/128 slot terisi aktif.")
            for s in active_slots[:10]:
                imp = model.bank.mem_importance[s].item()
                conf = model.bank.mem_confidence[s].item()
                acc = model.bank.mem_access_count[s].item()
                print(f"  Slot #{s.item()}: Importance={imp:.2f} | Conf={conf:.2f} | Access={acc}")
            print()
            continue

        with torch.no_grad():
            # 1. TULIS: Catat representasi fakta ke slot TinyMemoryBank menggunakan Konsep N-Token Representasi
            prompt_write = f"User: {user_input}\nAI:"
            c_inputs = tokenizer(prompt_write, return_tensors="pt").to(device)
            c_out = model(input_ids=c_inputs["input_ids"])
            c_len = c_inputs["input_ids"].shape[1]

            pfx_ids = tokenizer("User: ", return_tensors="pt")["input_ids"]
            pfx_len = pfx_ids.shape[1]
            end_idx = c_len - 2  # Menghindari token newline (\n) dan token 'AI:'

            if end_idx > pfx_len:
                u_hiddens = c_out["hidden_states"][0, pfx_len:end_idx]
                u_w_probs = c_out["write_probs"][0, pfx_len:end_idx]
                # Menulis sequence token ke memory bank per segmen N=8 (Opsi 1: Attention-Weighted Soft Pooling)
                model.write_memory_chunked(u_hiddens, u_w_probs, chunk_size=8)

            # 2. BACA & GENERASI: Token AI membaca slot TinyMemoryBank yang sudah diperbarui
            prompt = f"User: {user_input}\nAI:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            out = model(input_ids=inputs["input_ids"])
            read_prob = out["read_probs"][0, -1].item()

            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=35,
                temperature=0.3,
                top_k=40,
                top_p=0.85,
                repetition_penalty=1.25,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                stop_token_ids=[199]  # Stop segera pada karakter newline (\n)
            )

        new_tokens = outputs[0][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if "User:" in response:
            response = response.split("User:")[0].strip()
        if "\n" in response:
            response = response.split("\n")[0].strip()
        if not response:
            response = "Halo! Senang mengobrol denganmu. Ada yang bisa kubantu?"

        avg_w = u_w_probs.mean().item() if end_idx > pfx_len else 0.0
        print(f"\n[Turn {turn}] AI: {response}")
        print(f"      └─ [TinyMemoryBank]: Avg Write Prob={avg_w:.2f} | Read Prob={read_prob:.2f} | Active Slots={model.bank.active_count}/128\n")
        turn += 1

if __name__ == "__main__":
    main()
