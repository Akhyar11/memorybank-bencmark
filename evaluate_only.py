"""
evaluate_only.py – Load trained checkpoint and run evaluation without retraining.

Usage:
    python evaluate_only.py                          # evaluasi semua checkpoint di checkpoints/
    python evaluate_only.py --seed 42               # evaluasi hanya seed 42
    python evaluate_only.py --baseline memory_bank  # evaluasi baseline tertentu
    python evaluate_only.py --list                   # tampilkan daftar checkpoint tersedia
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.transformer_qa_model import TransformerQAModel
from models.tiny_memory_bank import TinyMemoryConfig
from dataset.text_dataset_loader import TextDataset
from evaluation.metrics import exact_match, batch_token_f1, recall_at_k, mean_reciprocal_rank


def collate_fn(batch):
    return torch.utils.data.dataloader.default_collate(batch)


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt


def evaluate_checkpoint(ckpt_path, dataset_dir, batch_size=32, max_test_samples=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = load_checkpoint(ckpt_path, device)
    seed      = ckpt.get('seed', 42)
    baseline  = ckpt.get('baseline', '?')
    epoch     = ckpt.get('epoch', '?')
    final_loss = ckpt.get('final_loss', float('nan'))

    print(f"\n{'='*65}")
    print(f"  Checkpoint : {os.path.basename(ckpt_path)}")
    print(f"  Baseline   : {baseline}")
    print(f"  Seed       : {seed}")
    print(f"  Trained Epochs : {epoch}")
    print(f"  Final Train Loss : {final_loss:.4f}")
    print(f"{'='*65}")

    test_csv = os.path.join(dataset_dir, 'test.csv')
    tokenizer_path = os.path.join(dataset_dir, 'tokenizer.json')

    if not os.path.exists(test_csv):
        print("[ERROR] test.csv tidak ditemukan. Jalankan generator dataset terlebih dahulu.")
        return None

    test_ds = TextDataset(test_csv, tokenizer_path, max_input_len=32, max_target_len=16, max_samples=max_test_samples)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    embed_dim   = 32
    memory_cap  = 128
    config = TinyMemoryConfig(memory_capacity=memory_cap, memory_dim=embed_dim, hidden_size=embed_dim)

    model = TransformerQAModel(
        config=config,
        vocab_size=test_ds.vocab_size,
        embed_dim=embed_dim,
        num_layers=1,
        num_heads=2,
        ff_dim=216,
        pad_id=test_ds.pad_id,
        bos_id=test_ds.bos_id,
        eos_id=test_ds.eos_id,
    ).to(device)

    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Determine mode from baseline name
    bname = baseline.lower()
    if 'bank' in bname:
        mode = 'bank'
    elif 'nn' in bname:
        mode = 'nn'
    else:
        mode = 'none'

    ems, f1s, r1s, r5s, mrrs = [], [], [], [], []
    eval_nn_keys = torch.zeros(memory_cap, embed_dim, device=device)
    eval_nn_vals = torch.zeros(memory_cap, embed_dim, device=device)
    eval_nn_write_idx = 0
    fact_to_slot = {}
    slot_to_fact = {}

    if mode == 'bank':
        model.bank.load_memory_state(model.bank.empty_memory_state())

    with torch.no_grad():
        for batch in test_loader:
            write_ids  = batch['write_ids'].to(device)
            write_mask = batch['write_mask'].to(device)
            query_ids  = batch['query_ids'].to(device)
            query_mask = batch['query_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            b_size     = write_ids.size(0)
            batch_fact_ids = batch['fact_str_id']

            if mode == 'none':
                logits, sim, _, _ = model(query_ids, query_mask,
                    torch.ones(b_size, device=device), torch.zeros(b_size, device=device),
                    target_ids, memory_mode='none')
                preds = torch.argmax(logits, dim=-1)
                written_indices = [-1] * b_size

            elif mode == 'nn':
                (k_proj, v_proj), _ = model.write_only(write_ids, write_mask,
                    torch.ones(b_size, device=device), torch.ones(b_size, device=device), memory_mode='nn')
                written_indices = []
                for i in range(b_size):
                    idx = (eval_nn_write_idx + i) % memory_cap
                    eval_nn_keys[idx] = k_proj[i]
                    eval_nn_vals[idx] = v_proj[i]
                    written_indices.append(idx)
                eval_nn_write_idx = (eval_nn_write_idx + b_size) % memory_cap
                logits, sim, _, _ = model(query_ids, query_mask,
                    torch.ones(b_size, device=device), torch.zeros(b_size, device=device),
                    target_ids, memory_mode='nn',
                    fact_store=(eval_nn_keys, eval_nn_vals))
                preds = torch.argmax(logits, dim=-1)

            else:
                _, written_indices = model.write_only(write_ids, write_mask,
                    torch.ones(b_size, device=device), torch.ones(b_size, device=device), memory_mode='bank')
                logits, sim, _, _ = model(query_ids, query_mask,
                    torch.ones(b_size, device=device), torch.zeros(b_size, device=device),
                    target_ids, memory_mode='bank')
                preds = torch.argmax(logits, dim=-1)
                written_indices = written_indices.cpu().numpy()

            preds = preds.cpu().numpy()
            sim   = sim.cpu().numpy()
            em = exact_match(preds, target_ids.cpu().numpy(), pad_id=test_ds.pad_id)
            f1 = batch_token_f1(preds, target_ids.cpu().numpy(), pad_id=test_ds.pad_id)
            ems.append(em)
            f1s.append(f1)

            if mode != 'none':
                for i in range(b_size):
                    fid  = batch_fact_ids[i]
                    slot = int(written_indices[i])
                    old  = slot_to_fact.get(slot)
                    if old and old != fid:
                        fact_to_slot[old] = None
                    slot_to_fact[slot] = fid
                    fact_to_slot[fid]  = slot

                for i in range(b_size):
                    fid    = batch_fact_ids[i]
                    gt_idx = fact_to_slot.get(fid)
                    if gt_idx is not None:
                        rd = recall_at_k(np.array(sim[i]), gt_idx, k_values=[1, 5])
                        r1s.append(rd[1])
                        r5s.append(rd[5])
                        mrrs.append(mean_reciprocal_rank(np.array(sim[i]), gt_idx))
                    else:
                        r1s.append(0.0)
                        r5s.append(0.0)
                        mrrs.append(0.0)

    print(f"\n  HASIL EVALUASI ({baseline}, Seed {seed}):")
    print(f"  {'─'*55}")
    print(f"  Exact Match (EM)   : {np.mean(ems)*100:.2f}%")
    print(f"  Token F1           : {np.mean(f1s)*100:.2f}%")
    if mode != 'none':
        print(f"  Recall@1           : {np.mean(r1s)*100:.2f}%")
        print(f"  Recall@5           : {np.mean(r5s)*100:.2f}%")
        print(f"  MRR                : {np.mean(mrrs):.4f}")
    print(f"  {'─'*55}")

    return {
        'baseline': baseline, 'seed': seed,
        'em': np.mean(ems)*100, 'f1': np.mean(f1s)*100,
        'r1': np.mean(r1s)*100 if r1s else None,
        'r5': np.mean(r5s)*100 if r5s else None,
        'mrr': np.mean(mrrs) if mrrs else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved Memory Bank checkpoints")
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='Directory containing .pt checkpoints')
    parser.add_argument('--seed', type=int, default=None, help='Evaluate only this seed')
    parser.add_argument('--baseline', default=None, help='Filter by baseline name substring')
    parser.add_argument('--list', action='store_true', help='List available checkpoints')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_test_samples', type=int, default=None)
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    if not os.path.exists(ckpt_dir):
        print(f"[ERROR] Checkpoint directory '{ckpt_dir}' tidak ditemukan.")
        print("Jalankan training terlebih dahulu: python experiments/end_to_end_benchmark.py")
        return

    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    if not ckpt_files:
        print(f"[ERROR] Tidak ada checkpoint (.pt) di '{ckpt_dir}'.")
        return

    # Filter
    if args.seed is not None:
        ckpt_files = [f for f in ckpt_files if f'seed{args.seed}' in f]
    if args.baseline:
        ckpt_files = [f for f in ckpt_files if args.baseline.lower().replace(' ', '_') in f.lower()]

    if args.list:
        print(f"\nCheckpoint tersedia di '{ckpt_dir}':")
        for f in ckpt_files:
            path = os.path.join(ckpt_dir, f)
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {f}  ({size_mb:.2f} MB)")
        return

    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"=== EVALUATE ONLY MODE ===")
    print(f"Device    : {device}")
    print(f"Checkpoint dir : {os.path.abspath(ckpt_dir)}")
    print(f"Found {len(ckpt_files)} checkpoint(s)")

    all_results = []
    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        result = evaluate_checkpoint(ckpt_path, dataset_dir,
            batch_size=args.batch_size,
            max_test_samples=args.max_test_samples,
            device=device)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print("  RINGKASAN SEMUA BASELINE")
        print(f"{'='*65}")
        print(f"  {'Baseline':<20} {'Seed':>5} {'EM':>7} {'F1':>7} {'R@1':>7} {'R@5':>7} {'MRR':>8}")
        print(f"  {'-'*60}")
        for r in all_results:
            r1  = f"{r['r1']:.2f}%" if r['r1'] is not None else '   —'
            r5  = f"{r['r5']:.2f}%" if r['r5'] is not None else '   —'
            mrr = f"{r['mrr']:.4f}" if r['mrr'] is not None else '      —'
            print(f"  {r['baseline']:<20} {r['seed']:>5} {r['em']:>6.2f}% {r['f1']:>6.2f}% {r1:>7} {r5:>7} {mrr:>8}")


if __name__ == '__main__':
    main()
