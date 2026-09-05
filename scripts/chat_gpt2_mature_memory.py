import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt2_memory_model import GPT2MemoryModel
from models.tiny_memory_bank import TinyMemoryConfig


def load_checkpoint_if_exists(model: GPT2MemoryModel, checkpoint_path: str, device: torch.device) -> bool:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return False

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return True

    if "adapter_state_dict" in ckpt:
        model.load_state_dict(ckpt["adapter_state_dict"], strict=False)
        return True

    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=False)
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with GPT-2 + differentiable causal memory")
    parser.add_argument("--model_dir", type=str, default="gpt2-indo-instruct-tuned")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gpt2_causal_memory_best.pt")
    parser.add_argument("--max_new_tokens", type=int, default=48)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 68)
    print(" GPT-2 INDO + DIFFERENTIABLE CAUSAL MEMORY CHAT")
    print("=" * 68)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mem_config = TinyMemoryConfig(memory_capacity=128, memory_dim=768, hidden_size=768)
    model = GPT2MemoryModel(
        model_name_or_path=args.model_dir,
        memory_config=mem_config,
        freeze_backbone=True,
    ).to(device)

    loaded = load_checkpoint_if_exists(model, args.checkpoint, device)
    if loaded:
        print(f"Checkpoint loaded: {args.checkpoint}")
    else:
        print("Checkpoint tidak ditemukan, menggunakan bobot saat ini.")

    model.eval()

    print("-" * 68)
    print("Perintah: /slots, /decay, /reset, /exit")
    print("-" * 68)

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
            turn = 1
            print("[SISTEM]: Memory direset.")
            continue

        if user_input == "/decay":
            print("[SISTEM]: Memory decay tidak digunakan (arsitektur murni tanpa threshold/decay).")
            continue

        if user_input == "/slots":
            occ = model.bank.mem_occupancy
            usage = model.bank.mem_usage
            age = model.bank.mem_age
            occ_sum = occ.sum().item()
            occ_mean = occ.mean().item()
            usage_mean = usage.mean().item()
            print(f"[MEMORY]: Capacity {model.bank.config.memory_capacity} slots | Occupancy Sum: {occ_sum:.2f} | Occ Mean: {occ_mean:.3f} | Usage Mean: {usage_mean:.3f}")
            top_slots = torch.argsort(occ, descending=True)[:10]
            for s in top_slots:
                o_val = occ[s].item()
                u_val = usage[s].item()
                a_val = age[s].item()
                if o_val > 0.001 or u_val > 0.001:
                    print(f"  Slot {s.item():03d} | occ={o_val:.3f} usage={u_val:.3f} age={a_val:.1f}")
            continue

        with torch.no_grad():
            prompt = f"User: {user_input}\nAI:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                temperature=0.6,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        if "User:" in response:
            response = response.split("User:")[0].strip()
        if "\n" in response:
            response = response.split("\n")[0].strip()
        if not response:
            response = "Baik, saya siap membantu."

        diag = model.last_diagnostics
        print(f"\n[Turn {turn}] AI: {response}")
        print(
            f"  [Memory] occ_sum={diag.get('occupancy_sum', 0.0):.1f} "
            f"occ_mean={diag.get('occupancy_mean', 0.0):.3f} "
            f"usage_mean={diag.get('usage_mean', 0.0):.3f} "
            f"avg_write_gate={diag.get('avg_write_gate', 0.0):.3f}\n"
        )

        turn += 1


if __name__ == "__main__":
    main()
