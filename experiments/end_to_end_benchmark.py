"""
experiments/end_to_end_benchmark.py (Scientific PyTorch Version)

Evaluates TransformerQAModel across 3 rigorous baselines:
1. No Memory (Clean baseline: query encoder directly to decoder context, bypassing memory adapter)
2. NN Memory (True Key-Value Neural Network Memory with separate K and V representations)
3. Memory Bank (Full Persistent Memory Bank with Decay, Read, Write, and Multi-factor Scoring)

Scientific Rigor Guarantees:
- Identical deterministic batch sequences across all baselines (P0-05)
- True Key-Value NN baseline parity (P0-08)
- Clean No-Memory baseline without unused memory adapter parameters (P0-06)
- Retrieval evaluated using actual composite multi-factor memory bank scores (P1-02)
- Explicit pair-wise target fact resolution instead of string replacement (P1-01)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import collections
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from models.transformer_qa_model import TransformerQAModel
from models.tiny_memory_bank import TinyMemoryConfig
from dataset.text_dataset_loader import TextDataset
from evaluation.metrics import exact_match, batch_token_f1, recall_at_k, mean_reciprocal_rank


def cross_entropy_loss(logits, targets, pad_id):
    """Sequence cross-entropy loss ignoring pad_id tokens."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=pad_id,
        reduction='mean'
    )


def collate_fn(batch):
    return torch.utils.data.dataloader.default_collate(batch)


def run_benchmark(
    num_epochs=100,
    seeds=(42, 43, 44),
    batch_size=8,
    max_train_samples=None,
    max_test_samples=128,
    embed_dim=32,
    num_layers=1,
    num_heads=2,
    ff_dim=216,
    memory_capacity=128
):
    print("===========================================")
    print("      END-TO-END MEMORY BENCHMARK          ")
    print("===========================================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}")
    print("===========================================")

    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    train_csv = os.path.join(dataset_dir, 'train.csv')
    test_csv = os.path.join(dataset_dir, 'test.csv')
    tokenizer_path = os.path.join(dataset_dir, 'tokenizer.json')

    if not os.path.exists(test_csv):
        print("Dataset not found. Please generate it first.")
        return

    train_dataset = TextDataset(train_csv, tokenizer_path, max_input_len=32, max_target_len=16, max_samples=max_train_samples)
    test_dataset = TextDataset(test_csv, tokenizer_path, max_input_len=32, max_target_len=16, max_samples=max_test_samples)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    config = TinyMemoryConfig(memory_capacity=memory_capacity, memory_dim=embed_dim, hidden_size=embed_dim)
    modes = {"No Memory": "none", "NN Memory": "nn", "Memory Bank": "bank"}

    results = {name: collections.defaultdict(list) for name in modes.keys()}

    for seed in seeds:
        print(f"\n[Seed {seed}] Preparing Deterministic Data Pipeline...")

        # Precompute exact deterministic batch sequences for all epochs (P0-05)
        # Guarantees that No Memory, NN Memory, and Memory Bank see the exact same batches in the exact same order
        epoch_batches = []
        for ep in range(num_epochs):
            g = torch.Generator()
            g.manual_seed(seed * 10000 + ep)
            sampler = RandomSampler(train_dataset, generator=g)
            loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
            epoch_batches.append(list(loader))

        for name, mode in modes.items():
            print(f"  --- Training & Evaluating {name} (Seed {seed}) ---")

            # Reset random seed before model initialization
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            mdl = TransformerQAModel(
                config=config, vocab_size=train_dataset.vocab_size,
                embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, ff_dim=ff_dim,
                pad_id=train_dataset.pad_id, bos_id=train_dataset.bos_id, eos_id=train_dataset.eos_id
            ).to(device)

            param_count = sum(p.numel() for p in mdl.parameters())
            print(f"      Model Parameters: {param_count:,}")

            optimizer = torch.optim.AdamW(mdl.parameters(), lr=1e-3)

            # Separate key and value buffers for NN Baseline (P0-08)
            nn_keys = torch.zeros(config.memory_capacity, mdl.embed_dim, device=device)
            nn_vals = torch.zeros(config.memory_capacity, mdl.embed_dim, device=device)
            nn_write_idx = 0

            # ---------------------------------------------------------------
            # Training Loop
            # ---------------------------------------------------------------
            for epoch in range(num_epochs):
                start_time = time.time()
                mdl.train()
                losses = []
                batches = epoch_batches[epoch]

                for batch in batches:
                    write_ids = batch['write_ids'].to(device)
                    write_mask = batch['write_mask'].to(device)
                    query_ids = batch['query_ids'].to(device)
                    query_mask = batch['query_mask'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    b_size = write_ids.size(0)

                    optimizer.zero_grad()

                    if mode == 'none':
                        # P0-06: Clean No Memory baseline
                        logits, _, _, _ = mdl(
                            query_ids, query_mask,
                            torch.ones(b_size, device=device),
                            torch.zeros(b_size, device=device),
                            target_ids, memory_mode='none'
                        )
                        loss = cross_entropy_loss(logits, target_ids, train_dataset.pad_id)

                    elif mode == 'nn':
                        # P0-08: True Key-Value NN Memory
                        (k_proj, v_proj), _ = mdl.write_only(
                            write_ids, write_mask,
                            torch.ones(b_size, device=device),
                            torch.ones(b_size, device=device),
                            memory_mode='nn'
                        )
                        for i in range(b_size):
                            slot = (nn_write_idx + i) % config.memory_capacity
                            nn_keys[slot] = k_proj[i].detach()
                            nn_vals[slot] = v_proj[i].detach()

                        logits, _, _, _ = mdl(
                            query_ids, query_mask,
                            torch.ones(b_size, device=device),
                            torch.zeros(b_size, device=device),
                            target_ids, memory_mode='nn', fact_store=(nn_keys, nn_vals)
                        )
                        loss = cross_entropy_loss(logits, target_ids, train_dataset.pad_id)
                        nn_write_idx = (nn_write_idx + b_size) % config.memory_capacity

                    else:
                        # Memory Bank (P0-07: Episodic write + read)
                        mdl.write_only(
                            write_ids, write_mask,
                            torch.ones(b_size, device=device),
                            torch.ones(b_size, device=device),
                            memory_mode='bank'
                        )
                        logits, _, _, _ = mdl(
                            query_ids, query_mask,
                            torch.ones(b_size, device=device),
                            torch.zeros(b_size, device=device),
                            target_ids, memory_mode='bank'
                        )
                        loss = cross_entropy_loss(logits, target_ids, train_dataset.pad_id)

                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                end_time = time.time()
                epoch_time = end_time - start_time
                steps = max(len(losses), 1)
                step_time_ms = (epoch_time / steps) * 1000
                print(f"    Epoch {epoch+1:3d} Loss: {np.mean(losses):.4f} | Time: {epoch_time:.2f}s | {step_time_ms:.2f} ms/step")

            # ---------------------------------------------------------------
            # Evaluation Phase
            # ---------------------------------------------------------------
            mdl.eval()
            ems, f1s, r1s, mrrs = [], [], [], []

            # Reset memory state before evaluation (P0-01)
            if mode == 'bank':
                mdl.bank.load_memory_state(mdl.bank.empty_memory_state())

            eval_nn_keys = torch.zeros(config.memory_capacity, mdl.embed_dim, device=device)
            eval_nn_vals = torch.zeros(config.memory_capacity, mdl.embed_dim, device=device)
            eval_nn_write_idx = 0

            # Ground truth tracking maps
            fact_to_slot = {}
            slot_to_fact = {}

            with torch.no_grad():
                for batch in test_loader:
                    write_ids = batch['write_ids'].to(device)
                    write_mask = batch['write_mask'].to(device)
                    query_ids = batch['query_ids'].to(device)
                    query_mask = batch['query_mask'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    b_size = write_ids.size(0)

                    batch_fact_ids = batch['fact_str_id']

                    if mode == 'none':
                        logits, sim, _, _ = mdl(
                            query_ids, query_mask,
                            torch.ones(b_size, device=device),
                            torch.zeros(b_size, device=device),
                            target_ids, memory_mode='none'
                        )
                        preds = torch.argmax(logits, dim=-1)

                    elif mode == 'nn':
                        (k_proj, v_proj), _ = mdl.write_only(
                            write_ids, write_mask,
                            torch.ones(b_size, device=device),
                            torch.ones(b_size, device=device),
                            memory_mode='nn'
                        )
                        written_indices = []
                        for i in range(b_size):
                            idx = (eval_nn_write_idx + i) % config.memory_capacity
                            eval_nn_keys[idx] = k_proj[i]
                            eval_nn_vals[idx] = v_proj[i]
                            written_indices.append(idx)

                        logits, sim, _, _ = mdl(
                            query_ids, query_mask,
                            torch.ones(b_size, device=device),
                            torch.zeros(b_size, device=device),
                            target_ids, memory_mode='nn', fact_store=(eval_nn_keys, eval_nn_vals)
                        )
                        preds = torch.argmax(logits, dim=-1)
                        eval_nn_write_idx = (eval_nn_write_idx + b_size) % config.memory_capacity

                    else:
                        # Memory Bank
                        _, written_indices = mdl.write_only(
                            write_ids, write_mask,
                            torch.ones(b_size, device=device),
                            torch.ones(b_size, device=device),
                            memory_mode='bank'
                        )
                        # P1-02: sim contains actual multi-factor composite scores from MemoryBank
                        logits, sim, _, _ = mdl(
                            query_ids, query_mask,
                            torch.ones(b_size, device=device),
                            torch.zeros(b_size, device=device),
                            target_ids, memory_mode='bank'
                        )
                        preds = torch.argmax(logits, dim=-1)
                        written_indices = written_indices.cpu().numpy()

                    preds = preds.cpu().numpy()
                    targets = target_ids.cpu().numpy()
                    sim = sim.cpu().numpy()

                    em = exact_match(preds, targets, pad_id=test_dataset.pad_id)
                    f1 = batch_token_f1(preds, targets, pad_id=test_dataset.pad_id)

                    if mode != 'none':
                        # Update ground truth slot mappings
                        for i in range(b_size):
                            fid = batch_fact_ids[i]
                            slot = int(written_indices[i])

                            old_fact = slot_to_fact.get(slot)
                            if old_fact is not None and old_fact != fid:
                                fact_to_slot[old_fact] = None

                            slot_to_fact[slot] = fid
                            fact_to_slot[fid] = slot

                        # P1-01: Pair-wise explicit ground-truth fact ID resolution
                        for i in range(b_size):
                            expected_fid = batch_fact_ids[i]  # Target fact corresponding to this query
                            gt_idx = fact_to_slot.get(expected_fid, None)

                            if gt_idx is not None:
                                r1 = recall_at_k(np.array(sim[i]), gt_idx, k_values=[1])[1]
                                mrr = mean_reciprocal_rank(np.array(sim[i]), gt_idx)
                            else:
                                r1 = 0.0
                                mrr = 0.0

                            r1s.append(r1)
                            mrrs.append(mrr)

                    ems.append(em)
                    f1s.append(f1)

            results[name]['loss'].append(float(np.mean(losses)))
            results[name]['em'].append(float(np.mean(ems) * 100))
            results[name]['f1'].append(float(np.mean(f1s) * 100))
            if mode != 'none':
                results[name]['r1'].append(float(np.mean(r1s) * 100))
                results[name]['mrr'].append(float(np.mean(mrrs)))

    print("\n===========================================")
    print("             FINAL BENCHMARK               ")
    print("===========================================")
    for name in modes.keys():
        print(f"\n{name} (Averaged over {len(seeds)} seeds):")
        print(f"  Final Train Loss: {np.mean(results[name]['loss']):.4f} ± {np.std(results[name]['loss']):.4f}")
        print(f"  Exact Match:      {np.mean(results[name]['em']):.2f}% ± {np.std(results[name]['em']):.2f}%")
        print(f"  Token F1:         {np.mean(results[name]['f1']):.2f}% ± {np.std(results[name]['f1']):.2f}%")
        if name != "No Memory":
            print(f"  Recall@1:         {np.mean(results[name]['r1']):.2f}% ± {np.std(results[name]['r1']):.2f}%")
            print(f"  MRR:              {np.mean(results[name]['mrr']):.4f} ± {np.std(results[name]['mrr']):.4f}")

    return results


if __name__ == '__main__':
    run_benchmark()
