"""
experiments/conversational_ntp_benchmark.py – Pure Decoder-Only Conversational NTP Benchmark.

Evaluates:
- No Memory (pure causal decoder baseline)
- NN Memory (independent Key-Value nearest neighbor baseline)
- Memory Bank (locked episodic Memory Bank)

Trained with causal Next-Token Prediction (NTP) on ChatML multi-turn conversational dataset.
Multi-seed support (42, 43, 44) with mean ± std reporting across:
- EM, Token F1
- Recall@1, Recall@5, MRR
- Write rate, Duplicate rate, Replacement rate, Occupancy
- Old value suppression rate
- Causal memory intervention rate
"""
import os
import sys
import time
import json
import collections
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tiny_memory_bank import TinyMemoryConfig
from models.decoder_only_memory_model import DecoderOnlyMemoryLM
from dataset.conversation_dataset import ConversationDataset, conversation_collate_fn, get_or_create_tokenizer
from evaluation.conversational_evaluator import ConversationalEvaluator


def run_conversational_benchmark(
    train_jsonl: str = "dataset/conversations_100M_train.jsonl",
    val_jsonl: str = "dataset/conversations_100M_val.jsonl",
    test_jsonl: str = "dataset/conversations_100M_test.jsonl",
    tokenizer_path: str = "dataset/tokenizer.json",
    seeds: Tuple[int, ...] = (42, 43, 44),
    num_epochs: int = 5,
    max_steps_per_epoch: Optional[int] = None,
    batch_size: int = 16,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = 200,
    embed_dim: int = 32,
    num_layers: int = 1,
    num_heads: int = 2,
    ff_dim: int = 216,
    learning_rate: float = 1e-3,
    checkpoint_dir: Optional[str] = None,
    write_threshold: float = 0.5,
    mode_filter: str = "all"
) -> Dict[str, Any]:
    print("==================================================================")
    print("       PURE DECODER-ONLY CONVERSATIONAL NTP BENCHMARK             ")
    print("==================================================================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE         : {device}")
    print(f"TRAIN DATASET  : {train_jsonl}")
    print(f"TEST DATASET   : {test_jsonl}")
    print(f"SEEDS          : {list(seeds)}")
    print(f"BUDGET         : {num_epochs} epochs x {max_steps_per_epoch} steps/epoch (Batch: {batch_size})")
    print("==================================================================")

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints_decoder_only')
    os.makedirs(checkpoint_dir, exist_ok=True)

    tok = get_or_create_tokenizer(tokenizer_path)
    vocab_size = tok.get_vocab_size()

    train_ds = ConversationDataset(train_jsonl, tokenizer_path=tokenizer_path, max_samples=max_train_samples)
    test_ds = ConversationDataset(test_jsonl, tokenizer_path=tokenizer_path, max_samples=max_test_samples)
    print(f"DATASET STATS  : Loaded {len(train_ds):,} train dialogues, {len(test_ds):,} test dialogues")

    # Load raw test items for logical evaluation
    test_items = []
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            test_items.append(json.loads(line))
            if max_test_samples is not None and len(test_items) >= max_test_samples:
                break

    all_modes = {
        "No Memory": "none",
        "NN Memory": "nn",
        "Memory Bank": "bank"
    }
    if mode_filter.lower() in ("bank", "memory_bank", "memorybank"):
        modes = {"Memory Bank": "bank"}
    elif mode_filter.lower() in ("none", "no_memory"):
        modes = {"No Memory": "none"}
    elif mode_filter.lower() in ("nn", "nn_memory"):
        modes = {"NN Memory": "nn"}
    else:
        modes = all_modes

    results = {name: collections.defaultdict(list) for name in modes.keys()}

    for seed in seeds:
        print(f"\n[Seed {seed}] Preparing Deterministic Conversational Data Pipeline...")

        # Precompute exact deterministic batch sequences
        epoch_batches = []
        for ep in range(num_epochs):
            g = torch.Generator()
            g.manual_seed(seed * 10000 + ep)
            sampler = RandomSampler(train_ds, generator=g)
            loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=sampler,
                collate_fn=lambda b: conversation_collate_fn(b, pad_id=train_ds.pad_id)
            )
            # Cap at max_steps_per_epoch
            batches = []
            for b_idx, batch in enumerate(loader):
                batches.append(batch)
                if max_steps_per_epoch is not None and b_idx + 1 >= max_steps_per_epoch:
                    break
            epoch_batches.append(batches)

        for name, mode in modes.items():
            print(f"\n  --- Training & Evaluating {name} (Seed {seed}) ---")

            # Deterministic initialization for all 3 baselines
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            cfg = TinyMemoryConfig(
                memory_capacity=128, memory_dim=embed_dim, hidden_size=embed_dim,
                memory_write_threshold=write_threshold, mem_alpha=2.0, mem_reinforcement_rate=0.01
            )
            model = DecoderOnlyMemoryLM(
                config=cfg, vocab_size=vocab_size,
                embed_dim=embed_dim, num_layers=num_layers,
                num_heads=num_heads, ff_dim=ff_dim,
                pad_id=train_ds.pad_id, bos_id=train_ds.bos_id, eos_id=train_ds.eos_id
            ).to(device)

            if seed == seeds[0] and list(modes.values())[0] == mode:
                print(f"      Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

            ckpt_path = os.path.join(checkpoint_dir, f"seed{seed}_{name.lower().replace(' ', '_')}.pt")
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

            total_train_tokens = 0
            final_loss = 0.0

            # Training Loop (NTP with Causal LM Loss)
            model.train()
            for ep in range(num_epochs):
                ep_loss = 0.0
                ep_batches = epoch_batches[ep]
                n_batches = len(ep_batches)

                pbar = tqdm(
                    ep_batches,
                    desc=f"    Epoch {ep+1:2d}/{num_epochs:2d}",
                    leave=True,
                    unit="batch",
                    dynamic_ncols=True
                )
                for step_idx, batch in enumerate(pbar):
                    input_ids = batch['input_ids'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    b_size = input_ids.size(0)

                    optimizer.zero_grad()
                    dialogue_losses = []

                    for b_i in range(b_size):
                        curr_input = input_ids[b_i:b_i+1, :batch['seq_lens'][b_i]]
                        curr_target = target_ids[b_i:b_i+1, :batch['seq_lens'][b_i]]
                        t_ends = batch['turn_end_indices'][b_i]
                        facts = batch['facts'][b_i]
                        t_recall = batch['target_recall'][b_i]

                        out_d = model.forward_dialogue_autoregressive(
                            input_ids=curr_input,
                            target_ids=curr_target,
                            turn_end_indices=t_ends,
                            facts=facts,
                            target_recall=t_recall,
                            memory_mode=mode,
                            write_threshold=write_threshold
                        )
                        dialogue_losses.append(out_d['loss'])

                    loss = torch.stack(dialogue_losses).mean()
                    loss.backward()
                    optimizer.step()

                    ep_loss += loss.item()
                    total_train_tokens += int(batch['loss_mask'].sum().item())

                    avg_so_far = ep_loss / (step_idx + 1)
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "avg": f"{avg_so_far:.4f}"
                    })

                final_loss = ep_loss / max(n_batches, 1)

            # Save checkpoint with explicit versioning
            torch.save({
                'architecture': 'decoder_only_memory_bank',
                'version': '2.0',
                'baseline': name,
                'seed': seed,
                'model': model.state_dict(),
                'final_loss': final_loss,
                'total_train_tokens': total_train_tokens
            }, ckpt_path)

            # Evaluation on test conversations
            evaluator = ConversationalEvaluator(model, tok, device=device)
            eval_res = evaluator.evaluate_dataset(test_items, memory_mode=mode, write_threshold=write_threshold)

            results[name]['loss'].append(final_loss)
            results[name]['em'].append(eval_res['em'])
            results[name]['f1'].append(eval_res['f1'])
            results[name]['r1'].append(eval_res['r1'])
            results[name]['r5'].append(eval_res['r5'])
            results[name]['mrr'].append(eval_res['mrr'])
            results[name]['write_rate'].append(eval_res['write_rate'])
            results[name]['occupancy'].append(eval_res['occupancy'])
            results[name]['suppression'].append(eval_res['suppression_rate'])
            results[name]['causal'].append(eval_res['causal_intervention_rate'])
            results[name]['target_prob_increased'].append(eval_res.get('target_prob_increased_rate', 0.0))

            print(f"      [Eval {name} S{seed}] EM: {eval_res['em']:.2f}% | F1: {eval_res['f1']:.2f}% | "
                  f"R@1: {eval_res['r1']:.2f}% | R@5: {eval_res['r5']:.2f}% | MRR: {eval_res['mrr']:.4f}")

    # Summary Report
    print("\n" + "=" * 72)
    print("        FINAL PURE DECODER-ONLY BENCHMARK RESULTS (3 SEEDS)      ")
    print("=" * 72)

    for name in modes.keys():
        r = results[name]
        print(f"\n{name} (Averaged over {len(seeds)} seeds):")
        print(f"  Final Train Loss : {np.mean(r['loss']):.4f} ± {np.std(r['loss']):.4f}")
        print(f"  Exact Match      : {np.mean(r['em']):.2f}% ± {np.std(r['em']):.2f}%")
        print(f"  Token F1         : {np.mean(r['f1']):.2f}% ± {np.std(r['f1']):.2f}%")
        if name != "No Memory":
            print(f"  Recall@1         : {np.mean(r['r1']):.2f}% ± {np.std(r['r1']):.2f}%")
            print(f"  Recall@5         : {np.mean(r['r5']):.2f}% ± {np.std(r['r5']):.2f}%")
            print(f"  MRR              : {np.mean(r['mrr']):.4f} ± {np.std(r['mrr']):.4f}")
        if name == "Memory Bank":
            print(f"  Write Rate       : {np.mean(r['write_rate']):.2f}% ± {np.std(r['write_rate']):.2f}%")
            print(f"  Memory Occupancy : {np.mean(r['occupancy']):.2f}% ± {np.std(r['occupancy']):.2f}%")
            print(f"  Old Suppression  : {np.mean(r['suppression']):.2f}% ± {np.std(r['suppression']):.2f}%")
            print(f"  Causal Action    : {np.mean(r['causal']):.2f}% ± {np.std(r['causal']):.2f}%")
            print(f"  Target Prob Inc. : {np.mean(r['target_prob_increased']):.2f}% ± {np.std(r['target_prob_increased']):.2f}%")

    print("\n" + "=" * 72)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure Decoder-Only Conversational NTP Benchmark")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="Max steps per epoch (default: None, trains on all batches)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Max train dialogues to load (default: None, loads entire train file)")
    parser.add_argument("--max_test_samples", type=int, default=200,
                        help="Max test dialogues for evaluation (default: 200)")
    parser.add_argument("--write_threshold", type=float, default=0.6)
    parser.add_argument("--mode", type=str, default="all", choices=["all", "bank", "none", "nn"],
                        help="Model mode to train/evaluate: 'bank' (Memory Bank only), 'none', 'nn', or 'all'")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Random seeds to run (e.g. --seeds 42 or --seeds 42 43 44)")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "5m"],
                        help="Preset model size: 'tiny' (~155K params) or '5m' (~5M params, 1:20 Chinchilla optimal for 100M tokens)")
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--ff_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train_jsonl", type=str, default="dataset/conversations_100M_train.jsonl")
    parser.add_argument("--test_jsonl", type=str, default="dataset/conversations_100M_test.jsonl")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    args = parser.parse_args()

    # Preset configurations
    if args.model_size == "5m":
        embed_dim = args.embed_dim or 256
        num_layers = args.num_layers or 4
        num_heads = args.num_heads or 8
        ff_dim = args.ff_dim or 1024
    else:  # tiny
        embed_dim = args.embed_dim or 32
        num_layers = args.num_layers or 1
        num_heads = args.num_heads or 2
        ff_dim = args.ff_dim or 216

    run_conversational_benchmark(
        train_jsonl=args.train_jsonl,
        test_jsonl=args.test_jsonl,
        seeds=tuple(args.seeds),
        num_epochs=args.epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        learning_rate=args.lr,
        write_threshold=args.write_threshold,
        mode_filter=args.mode,
        checkpoint_dir=args.checkpoint_dir
    )
