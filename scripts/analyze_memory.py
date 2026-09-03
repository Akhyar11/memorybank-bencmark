"""
scripts/analyze_memory.py – Visual Analysis of Memory Slot & Read Distributions.

Analyzes:
1. Slot State & Occupancy Distribution (ACTIVE, DORMANT, EXPIRED).
2. Metadata Distribution (Importance, Confidence, Age/Recency).
3. Slot Read Distribution (Access counts, Hot vs Cold slots, ASCII distribution chart).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from models.tiny_memory_bank import (
    TinyMemoryBank, TinyMemoryConfig,
    STATE_EXPIRED, STATE_ACTIVE, STATE_DORMANT
)
from dataset.text_dataset_loader import TextDataset
from torch.utils.data import DataLoader
from models.transformer_qa_model import TransformerQAModel


def render_ascii_histogram(data, bins=10, width=40, title=""):
    """Render a clean text-based ASCII histogram."""
    data = np.array(data)
    if len(data) == 0:
        return "No data available."
    counts, bin_edges = np.histogram(data, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1

    lines = []
    if title:
        lines.append(f"  --- {title} ---")
    for i in range(len(counts)):
        low = bin_edges[i]
        high = bin_edges[i+1]
        bar_len = int((counts[i] / max_count) * width)
        bar = "█" * bar_len
        lines.append(f"  [{low:6.2f} - {high:6.2f}] : {bar:<{width}} ({counts[i]:>3d})")
    return "\n".join(lines)


def render_slot_grid(access_counts, capacity=128, cols=16):
    """Render a 2D grid showing read activity across memory slots."""
    lines = []
    lines.append(f"  Slot Grid (Read Access Intensity across {capacity} slots):")
    lines.append("  '.' = 0 reads | '1-9' = 1-9 reads | '+' = 10+ reads | '*' = 20+ reads")
    lines.append("  " + "-" * (cols * 2 + 1))

    for r in range(0, capacity, cols):
        row_str = f"  [{r:3d}-{r+cols-1:3d}] | "
        for c in range(cols):
            idx = r + c
            if idx < capacity:
                cnt = access_counts[idx]
                if cnt == 0:
                    ch = "."
                elif 1 <= cnt <= 9:
                    ch = str(cnt)
                elif 10 <= cnt < 20:
                    ch = "+"
                else:
                    ch = "*"
                row_str += ch + " "
        lines.append(row_str + "|")
    lines.append("  " + "-" * (cols * 2 + 1))
    return "\n".join(lines)


def analyze_memory_distribution(
    dataset_csv="dataset/train.csv",
    tokenizer_path="dataset/tokenizer.json",
    capacity=128,
    dim=32,
    num_samples=150
):
    print("=" * 70)
    print("       MEMORY BANK: SLOT & READ DISTRIBUTION ANALYSIS")
    print("=" * 70)

    config = TinyMemoryConfig(
        memory_capacity=capacity,
        memory_dim=dim,
        hidden_size=dim,
        memory_top_k=4,
        mem_decay_rate=0.01,
        mem_reinforcement_rate=0.05,
        memory_write_threshold=0.85
    )

    if not os.path.exists(dataset_csv) or not os.path.exists(tokenizer_path):
        print("Dataset not found, running with synthetic stream...")
        # Synthetic run
        bank = TinyMemoryBank(config=config)
        bank.load_memory_state(bank.empty_memory_state())
        g = torch.Generator().manual_seed(42)
        for i in range(num_samples):
            h = torch.randn(1, dim, generator=g)
            bank.write(h, torch.ones(1), torch.ones(1))
            if i % 3 == 0:
                bank.global_step[0] += 1
            if i > 10 and i % 2 == 0:
                q = torch.randn(1, dim, generator=g)
                bank.read(q)
    else:
        # Real QA dataset stream
        ds = TextDataset(dataset_csv, tokenizer_path, max_samples=num_samples)
        loader = DataLoader(ds, batch_size=16, shuffle=False)

        model = TransformerQAModel(
            config=config,
            vocab_size=ds.vocab_size,
            embed_dim=dim,
            num_layers=1,
            num_heads=2,
            ff_dim=216,
            pad_id=ds.pad_id,
            bos_id=ds.bos_id,
            eos_id=ds.eos_id
        )
        model.eval()

        with torch.no_grad():
            for batch in loader:
                w_ids = batch['write_ids']
                w_mask = batch['write_mask']
                q_ids = batch['query_ids']
                q_mask = batch['query_mask']
                t_ids = batch['target_ids']
                b = w_ids.size(0)

                # Write facts to bank
                model.write_only(w_ids, w_mask, torch.ones(b), torch.ones(b), memory_mode='bank')
                model.bank.global_step[0] += 1

                # Query bank
                model(q_ids, q_mask, torch.ones(b), torch.zeros(b), t_ids, memory_mode='bank')

        bank = model.bank

    # Extract state tensors
    state = bank.mem_state.cpu().numpy()
    importance = bank.mem_importance.cpu().numpy()
    confidence = bank.mem_confidence.cpu().numpy()
    created = bank.mem_created_at.cpu().numpy()
    last_access = bank.mem_last_access.cpu().numpy()
    access_count = bank.mem_access_count.cpu().numpy()
    current_step = bank.global_step.item()

    # 1. SLOT OCCUPANCY DISTRIBUTION
    n_active = int(np.sum(state == STATE_ACTIVE))
    n_dormant = int(np.sum(state == STATE_DORMANT))
    n_expired = int(np.sum(state == STATE_EXPIRED))

    print(f"\n[1] DISTRIBUSI STATUS SLOT MEMORI (Kapasitas: {capacity} Slot)")
    print("-" * 70)
    print(f"  • ACTIVE  (Aktif & Siap Diakses) : {n_active:3d} slot ({n_active/capacity*100:5.1f}%)")
    print(f"  • DORMANT (Tertidur / Meluruh)   : {n_dormant:3d} slot ({n_dormant/capacity*100:5.1f}%)")
    print(f"  • EXPIRED (Kosong / Siap Ganti)  : {n_expired:3d} slot ({n_expired/capacity*100:5.1f}%)")

    # 2. METADATA DISTRIBUTION
    active_mask = (state == STATE_ACTIVE)
    if np.any(active_mask):
        active_imp = importance[active_mask]
        active_conf = confidence[active_mask]
        active_age = current_step - created[active_mask]

        print(f"\n[2] STATISTIK METADATA SLOT AKTIF")
        print("-" * 70)
        print(f"  • Importance : Mean={np.mean(active_imp):.3f} | Min={np.min(active_imp):.3f} | Max={np.max(active_imp):.3f} | Std={np.std(active_imp):.3f}")
        print(f"  • Confidence : Mean={np.mean(active_conf):.3f} | Min={np.min(active_conf):.3f} | Max={np.max(active_conf):.3f} | Std={np.std(active_conf):.3f}")
        print(f"  • Umur Fakta : Mean={np.mean(active_age):.1f} step | Min={np.min(active_age)} | Max={np.max(active_age)}")

    # 3. READ / ACCESS DISTRIBUTION
    print(f"\n[3] DISTRIBUSI READ / AKSES SLOT MEMORI")
    print("-" * 70)
    total_reads = int(np.sum(access_count))
    unaccessed = int(np.sum(access_count == 0))
    hot_slots = np.where(access_count >= 5)[0]
    cold_slots = np.where((access_count > 0) & (access_count < 3))[0]

    print(f"  • Total Pembacaan (Read Ops)   : {total_reads} kali")
    print(f"  • Rata-rata Akses per Slot     : {np.mean(access_count):.2f} kali (Max: {np.max(access_count)})")
    print(f"  • Slot Belum Pernah Dibaca (0) : {unaccessed:3d} slot ({unaccessed/capacity*100:5.1f}%)")
    print(f"  • Hot Slots (≥ 5 kali dibaca)  : {len(hot_slots):3d} slot {list(hot_slots[:8])}...")
    print(f"  • Cold Slots (1-2 kali dibaca) : {len(cold_slots):3d} slot")

    print("\n[4] PETA SEBARAN AKSES READ PER SLOT (Visual Grid)")
    print(render_slot_grid(access_count, capacity=capacity, cols=16))

    print("\n[5] HISTOGRAM FREKUENSI AKSES READ:")
    print(render_ascii_histogram(access_count, bins=6, width=35, title="Frekuensi Akses Pembacaan"))
    print("=" * 70)


if __name__ == '__main__':
    analyze_memory_distribution()
