#!/usr/bin/env python3
"""
scripts/chat_cli.py - Interactive CLI Chat with Decoder-Only Memory Bank LM.
Allows users to chat directly with the trained model in real time,
viewing episodic memory writes, active slots, and retrieval scores.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tiny_memory_bank import TinyMemoryConfig
from models.decoder_only_memory_model import DecoderOnlyMemoryLM
from dataset.conversation_dataset import get_or_create_tokenizer

def decode_clean(tok, token_ids):
    special_ids = {0, 1, 2, 3, tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")}
    tokens = [tok.id_to_token(tid) for tid in token_ids if tid not in special_ids and tok.id_to_token(tid) is not None]
    return "".join(tokens).replace("▁", " ").replace("  ", " ").strip()

def main():
    ckpt_path = "checkpoints_decoder_only/seed42_memory_bank.pt"
    tokenizer_path = "dataset/tokenizer.json"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 70)
    print("      MEMORIS CHAT CLI - INTERACTIVE MEMORY BANK LM")
    print("=" * 70)
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Hardware Device   : {device}")

    tok = get_or_create_tokenizer(tokenizer_path)
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    m_args = ckpt.get("model_args", {})

    cfg = TinyMemoryConfig(
        memory_capacity=512,
        memory_dim=m_args.get("embed_dim", 256),
        hidden_size=m_args.get("embed_dim", 256),
        memory_write_threshold=0.5,
        mem_alpha=2.0,
        mem_reinforcement_rate=0.01
    )

    model = DecoderOnlyMemoryLM(
        config=cfg,
        vocab_size=m_args.get("vocab_size", 32003),
        embed_dim=m_args.get("embed_dim", 256),
        num_layers=m_args.get("num_layers", 4),
        num_heads=m_args.get("num_heads", 4),
        ff_dim=m_args.get("ff_dim", 1024),
        pad_id=tok.token_to_id("<PAD>") if tok.token_to_id("<PAD>") is not None else 0,
        bos_id=tok.token_to_id("<BOS>") if tok.token_to_id("<BOS>") is not None else 2,
        eos_id=tok.token_to_id("<EOS>") if tok.token_to_id("<EOS>") is not None else 3,
    ).to(device)

    buffer_keys = {k for k in ckpt['model'].keys() if 'bank.mem_' in k}
    filtered_state = {k: v for k, v in ckpt['model'].items() if k not in buffer_keys}
    model.load_state_dict(filtered_state, strict=False)
    model.reset_memory()
    model.eval()

    im_end_id = tok.token_to_id("<|im_end|>")

    print("\n" + "-" * 70)
    print("✓ Model siap diajak ngobrol!")
    print("Perintah khusus:")
    print("  /reset  : Mengosongkan memori & memulai obrolan baru")
    print("  /slots  : Melihat status slot memori yang sedang terisi")
    print("  /exit   : Keluar dari obrolan")
    print("-" * 70 + "\n")

    history_chatml = ""
    turn_counter = 1

    while True:
        try:
            user_input = input(f"\n[Turn {turn_counter}] Anda: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSampai jumpa!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["/exit", "/quit"]:
            print("Obrolan selesai. Sampai jumpa!")
            break

        if user_input.lower() == "/reset":
            model.reset_memory()
            history_chatml = ""
            turn_counter = 1
            print(">> [Memory Bank di-reset! Semua memori telah dikosongkan.] <<")
            continue

        if user_input.lower() == "/slots":
            active_c = model.bank.active_count
            print(f">> [Status Memory Bank: {active_c} dari {cfg.memory_capacity} slot terisi.] <<")
            continue

        # Format prompt
        user_turn_str = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        full_prompt = history_chatml + user_turn_str
        tokens = torch.tensor([tok.encode(full_prompt).ids], device=device)

        with torch.no_grad():
            gen_tokens, info = model.generate(
                input_ids=tokens,
                max_new_tokens=60,
                memory_mode="bank",
                write_threshold=0.5,
                temperature=0.7,
                top_k=40,
                im_end_id=im_end_id
            )

        resp_text = decode_clean(tok, gen_tokens[0].tolist())
        print(f"\n[Turn {turn_counter}] Assistant: {resp_text}")

        # Tampilkan status Memory Bank
        wp = info.get('write_prob', 0.0)
        did_w = info.get('did_write', False)
        active_slots = info.get('memory_active', 0)
        print(f"      └─ [Memory Bank]: Write Prob={wp:.3f} | Did Write={'Ya (Fakta Disimpan)' if did_w else 'Tidak'} | Slots Terisi={active_slots}")

        # Update chat history
        history_chatml += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n{resp_text}<|im_end|>\n"
        turn_counter += 1

if __name__ == "__main__":
    main()
