"""
experiments/end_to_end_benchmark.py (PyTorch Version)

Evaluates TransformerQAModel in 3 modes:
1. No Memory
2. NN Memory (Simple Dot-Product)
3. Memory Bank (Full)
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import collections
import time

from models.transformer_qa_model import TransformerQAModel
from models.tiny_memory_bank import TinyMemoryConfig
from dataset.text_dataset_loader import get_dataloader
from evaluation.metrics import exact_match, batch_token_f1, recall_at_k, mean_reciprocal_rank

def cross_entropy_loss(logits, targets, pad_id):
    # logits: (batch, seq_len, vocab)
    # targets: (batch, seq_len)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_id, reduction='mean')
    return loss

def run_benchmark():
    print("===========================================")
    print("      END-TO-END MEMORY BENCHMARK          ")
    print("===========================================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PYTORCH DEVICE DETECTED: {device}")
    print("===========================================")
    
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    train_csv = os.path.join(dataset_dir, 'train.csv')
    test_csv = os.path.join(dataset_dir, 'test.csv')
    tokenizer_path = os.path.join(dataset_dir, 'tokenizer.json')
    
    if not os.path.exists(test_csv):
        print("Dataset not found. Please generate it first.")
        return
        
    batch_size = 8
    train_loader = get_dataloader(train_csv, tokenizer_path, batch_size, 32, 16, max_samples=None, shuffle=True)
    test_loader = get_dataloader(test_csv, tokenizer_path, batch_size, 32, 16, max_samples=128, shuffle=False)
    
    config = TinyMemoryConfig(memory_capacity=128, memory_dim=256, hidden_size=256)
    
    modes = {"No Memory": "none", "NN Memory": "nn", "Memory Bank": "bank"}
    num_epochs = 100
    seeds = [42, 43, 44]
    
    results = {name: collections.defaultdict(list) for name in modes.keys()}

    for seed in seeds:
        print(f"\n[Seed {seed}] Initializing Model Backbone...")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        for name, mode in modes.items():
            print(f"  --- Training & Evaluating {name} (Seed {seed}) ---")
            
            torch.manual_seed(seed)
            mdl = TransformerQAModel(
                config=config, vocab_size=train_loader.vocab_size, 
                embed_dim=256, num_layers=4, num_heads=4, 
                pad_id=train_loader.pad_id, bos_id=train_loader.bos_id, eos_id=train_loader.eos_id
            ).to(device)
            
            optimizer = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
            
            fact_store = torch.zeros(config.memory_capacity, config.hidden_size, device=device)
            write_idx = 0
            
            for epoch in range(num_epochs):
                start_time = time.time()
                mdl.train()
                losses = []
                
                for batch in train_loader:
                    write_ids = batch['write_ids'].to(device)
                    write_mask = batch['write_mask'].to(device)
                    query_ids = batch['query_ids'].to(device)
                    query_mask = batch['query_mask'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    
                    batch_size_cur = write_ids.size(0)
                    
                    optimizer.zero_grad()
                    
                    if mode == 'none':
                        logits, _, _, _ = mdl(
                            query_ids, query_mask, 
                            torch.ones(batch_size_cur, device=device), 
                            torch.zeros(batch_size_cur, device=device), 
                            target_ids, memory_mode='none'
                        )
                        loss = cross_entropy_loss(logits, target_ids, train_loader.pad_id)
                        
                    elif mode == 'nn':
                        h_eos_proj, _ = mdl.write_only(
                            write_ids, write_mask, 
                            torch.ones(batch_size_cur, device=device), 
                            torch.ones(batch_size_cur, device=device), 
                            memory_mode='nn'
                        )
                        # sequentially write to fact_store
                        for i in range(batch_size_cur):
                            fact_store[(write_idx + i) % config.memory_capacity] = h_eos_proj[i].detach()
                            
                        logits, _, _, _ = mdl(
                            query_ids, query_mask, 
                            torch.ones(batch_size_cur, device=device), 
                            torch.zeros(batch_size_cur, device=device), 
                            target_ids, memory_mode='nn', fact_store=fact_store
                        )
                        loss = cross_entropy_loss(logits, target_ids, train_loader.pad_id)
                        write_idx = (write_idx + batch_size_cur) % config.memory_capacity
                        
                    else:
                        # Memory Bank
                        # Write first
                        _, _ = mdl.write_only(
                            write_ids, write_mask, 
                            torch.ones(batch_size_cur, device=device), 
                            torch.ones(batch_size_cur, device=device), 
                            memory_mode='bank'
                        )
                        # Then query
                        logits, _, _, _ = mdl(
                            query_ids, query_mask, 
                            torch.ones(batch_size_cur, device=device), 
                            torch.zeros(batch_size_cur, device=device), 
                            target_ids, memory_mode='bank'
                        )
                        loss = cross_entropy_loss(logits, target_ids, train_loader.pad_id)
                        
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    
                end_time = time.time()
                epoch_time = end_time - start_time
                steps = max(len(losses), 1)
                step_time_ms = (epoch_time / steps) * 1000
                print(f"    Epoch {epoch+1:3d} Loss: {np.mean(losses):.4f} | Time: {epoch_time:.2f}s | {step_time_ms:.2f} ms/step")
                
            # Evaluation
            mdl.eval()
            ems, f1s, r1s, mrrs = [], [], [], []
            
            # Reset memory for evaluation
            if mode == 'bank':
                # Reset tiny memory bank state completely
                mdl.bank.load_memory_state(mdl.bank.empty_memory_state())
            
            eval_fact_store = torch.zeros(config.memory_capacity, config.hidden_size, device=device)
            eval_write_idx = 0
            
            fact_to_slot = {}
            slot_to_fact = {}
            
            with torch.no_grad():
                for batch in test_loader:
                    write_ids = batch['write_ids'].to(device)
                    write_mask = batch['write_mask'].to(device)
                    query_ids = batch['query_ids'].to(device)
                    query_mask = batch['query_mask'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    batch_size_cur = write_ids.size(0)
                    
                    if mode == 'none':
                        logits, sim, _, _ = mdl(
                            query_ids, query_mask, 
                            torch.ones(batch_size_cur, device=device), 
                            torch.zeros(batch_size_cur, device=device), 
                            target_ids, memory_mode='none'
                        )
                        preds = torch.argmax(logits, dim=-1)
                    elif mode == 'nn':
                        h_eos_proj, _ = mdl.write_only(
                            write_ids, write_mask, 
                            torch.ones(batch_size_cur, device=device), 
                            torch.ones(batch_size_cur, device=device), 
                            memory_mode='nn'
                        )
                        written_indices = []
                        for i in range(batch_size_cur):
                            idx = (eval_write_idx + i) % config.memory_capacity
                            eval_fact_store[idx] = h_eos_proj[i]
                            written_indices.append(idx)
                            
                        logits, sim, _, _ = mdl(
                            query_ids, query_mask, 
                            torch.ones(batch_size_cur, device=device), 
                            torch.zeros(batch_size_cur, device=device), 
                            target_ids, memory_mode='nn', fact_store=eval_fact_store
                        )
                        preds = torch.argmax(logits, dim=-1)
                        eval_write_idx = (eval_write_idx + batch_size_cur) % config.memory_capacity
                    else:
                        _, written_indices = mdl.write_only(
                            write_ids, write_mask, 
                            torch.ones(batch_size_cur, device=device), 
                            torch.ones(batch_size_cur, device=device), 
                            memory_mode='bank'
                        )
                        logits, sim, _, _ = mdl(
                            query_ids, query_mask, 
                            torch.ones(batch_size_cur, device=device), 
                            torch.zeros(batch_size_cur, device=device), 
                            target_ids, memory_mode='bank'
                        )
                        preds = torch.argmax(logits, dim=-1)
                        written_indices = written_indices.cpu().numpy()

                    preds = preds.cpu().numpy()
                    targets = target_ids.cpu().numpy()
                    sim = sim.cpu().numpy()
                    
                    batch_fact_ids = batch['fact_str_id']
                    batch_query_ids = batch['query_str_id']
                    
                    em = exact_match(preds, targets, pad_id=test_loader.pad_id)
                    f1 = batch_token_f1(preds, targets, pad_id=test_loader.pad_id)
                    
                    if mode != 'none':
                        for i in range(batch_size_cur):
                            fid = batch_fact_ids[i]
                            slot = int(written_indices[i])
                            
                            old_fact = slot_to_fact.get(slot)
                            if old_fact is not None and old_fact != fid:
                                fact_to_slot[old_fact] = None
                                
                            slot_to_fact[slot] = fid
                            fact_to_slot[fid] = slot
                            
                        for i in range(batch_size_cur):
                            qid = batch_query_ids[i]
                            expected_fid = qid.replace("_Q", "_F")
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
                    
            results[name]['loss'].append(np.mean(losses))
            results[name]['em'].append(np.mean(ems)*100)
            results[name]['f1'].append(np.mean(f1s)*100)
            if mode != 'none':
                results[name]['r1'].append(np.mean(r1s)*100)
                results[name]['mrr'].append(np.mean(mrrs))
                
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

if __name__ == '__main__':
    run_benchmark()
