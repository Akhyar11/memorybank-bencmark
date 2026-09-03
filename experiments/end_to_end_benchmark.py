"""
experiments/end_to_end_benchmark.py

Evaluates TransformerQAModel in 3 modes:
1. No Memory
2. NN Memory (Simple Dot-Product)
3. Memory Bank (Full)
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
import numpy as np
import collections
import time

from models.transformer_qa_model import TransformerQAModel
from models.tiny_memory_bank import TinyMemoryConfig
from dataset.text_dataset_loader import TextDataLoader
from evaluation.metrics import exact_match, batch_token_f1, recall_at_k, mean_reciprocal_rank

def create_initial_memory(config):
    return {
        'keys': jnp.zeros((config.memory_capacity, config.memory_dim)),
        'vals': jnp.zeros((config.memory_capacity, config.memory_dim)),
        'importance': jnp.zeros(config.memory_capacity),
        'confidence': jnp.zeros(config.memory_capacity),
        'created_at': jnp.zeros(config.memory_capacity, dtype=jnp.int32),
        'last_access': jnp.zeros(config.memory_capacity, dtype=jnp.int32),
        'access_count': jnp.zeros(config.memory_capacity, dtype=jnp.int32),
        'state': jnp.zeros(config.memory_capacity, dtype=jnp.int32),
        'global_step': jnp.zeros((), dtype=jnp.int32)
    }

def cross_entropy_loss(logits, targets, pad_id, vocab_size):
    one_hot = jax.nn.one_hot(targets, vocab_size)
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    mask = (targets != pad_id)
    return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

def run_benchmark():
    print("===========================================")
    print("      END-TO-END MEMORY BENCHMARK          ")
    print("===========================================")
    print(f"JAX DEVICES DETECTED: {jax.devices()}")
    print("===========================================")
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    train_csv = os.path.join(dataset_dir, 'train.csv')
    test_csv = os.path.join(dataset_dir, 'test.csv')
    tokenizer_path = os.path.join(dataset_dir, 'tokenizer.json')
    
    if not os.path.exists(test_csv):
        print("Dataset not found. Please generate it first.")
        return
        
    batch_size = 8
    # Train loader
    train_loader = TextDataLoader(train_csv, tokenizer_path, batch_size, 32, 16, max_samples=512)
    # Test loader
    loader = TextDataLoader(test_csv, tokenizer_path, batch_size, 32, 16, max_samples=128)
    
    config = TinyMemoryConfig(memory_capacity=128, memory_dim=256, hidden_size=256)
    
    # We use ONE single backbone
    mdl = TransformerQAModel(config=config, vocab_size=loader.vocab_size, embed_dim=256, num_layers=4, num_heads=4, pad_id=loader.pad_id, bos_id=loader.bos_id, eos_id=loader.eos_id)
    
    dummy_input = jnp.ones((batch_size, 32), dtype=jnp.int32)
    dummy_target = jnp.ones((batch_size, 16), dtype=jnp.int32)
    
    modes = {"No Memory": "none", "NN Memory": "nn", "Memory Bank": "bank"}
    num_epochs = 2
    seeds = [42, 43, 44]
    
    results = {name: collections.defaultdict(list) for name in modes.keys()}

    # Initialize optimizer
    tx = optax.adamw(learning_rate=1e-3)

    @jax.jit
    def train_epoch_none(params, opt_state, mem_state, batched_data, rng):
        def scan_step(carry, xs):
            p, opt = carry
            b, r = xs
            def loss_fn(p_inner):
                logits, _, _, _ = mdl.apply({'params': p_inner, 'memory': mem_state}, b['query_ids'], b['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), b['target_ids'], deterministic=False, memory_mode='none', rngs={'dropout': r})
                return cross_entropy_loss(logits, b['target_ids'], loader.pad_id, loader.vocab_size)
            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, opt = tx.update(grads, opt, p)
            p = optax.apply_updates(p, updates)
            return (p, opt), loss
        
        num_b = batched_data['query_ids'].shape[0]
        rngs = jax.random.split(rng, num_b)
        (params, opt_state), losses = jax.lax.scan(scan_step, (params, opt_state), (batched_data, rngs))
        return params, opt_state, mem_state, losses

    @jax.jit
    def train_epoch_nn(params, opt_state, mem_state, fact_store, write_idx, batched_data, rng):
        def scan_step(carry, xs):
            p, opt, fs, widx = carry
            b, r = xs
            def loss_fn(p_inner):
                h_eos_proj, _ = mdl.apply({'params': p_inner}, b['write_ids'], b['write_mask'], True, jnp.ones(batch_size), deterministic=False, memory_mode='none', method=mdl.write_only, rngs={'dropout': r})
                new_fs = jax.lax.dynamic_update_slice(fs, h_eos_proj, (widx, 0))
                logits, _, _, _ = mdl.apply({'params': p_inner, 'memory': mem_state}, b['query_ids'], b['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), b['target_ids'], deterministic=False, memory_mode='nn', fact_store=new_fs, rngs={'dropout': r})
                return cross_entropy_loss(logits, b['target_ids'], loader.pad_id, loader.vocab_size), new_fs
            (loss, fs), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
            updates, opt = tx.update(grads, opt, p)
            p = optax.apply_updates(p, updates)
            return (p, opt, fs, (widx + batch_size) % config.memory_capacity), loss
            
        num_b = batched_data['query_ids'].shape[0]
        rngs = jax.random.split(rng, num_b)
        (params, opt_state, fact_store, write_idx), losses = jax.lax.scan(scan_step, (params, opt_state, fact_store, write_idx), (batched_data, rngs))
        return params, opt_state, fact_store, write_idx, losses

    @jax.jit
    def train_epoch_bank(params, opt_state, mem_state, batched_data, rng):
        def scan_step(carry, xs):
            p, opt, m = carry
            b, r = xs
            def loss_fn(p_inner):
                _, updated_m = mdl.apply({'params': p_inner, 'memory': m}, b['write_ids'], b['write_mask'], jnp.ones(batch_size), jnp.ones(batch_size), True, method=mdl.write_only, mutable=['memory'], rngs={'dropout': r})
                logits, _, _, _ = mdl.apply({'params': p_inner, 'memory': updated_m['memory']}, b['query_ids'], b['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), b['target_ids'], deterministic=False, memory_mode='bank', rngs={'dropout': r}, mutable=['memory'])[0]
                return cross_entropy_loss(logits, b['target_ids'], loader.pad_id, loader.vocab_size), updated_m['memory']
            (loss, m), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
            updates, opt = tx.update(grads, opt, p)
            p = optax.apply_updates(p, updates)
            return (p, opt, m), loss
            
        num_b = batched_data['query_ids'].shape[0]
        rngs = jax.random.split(rng, num_b)
        (params, opt_state, mem_state), losses = jax.lax.scan(scan_step, (params, opt_state, mem_state), (batched_data, rngs))
        return params, opt_state, mem_state, losses
        
    @jax.jit
    def eval_step_none(params, mem_state, batch):
        logits, sim, _, _ = mdl.apply({'params': params, 'memory': mem_state}, batch['query_ids'], batch['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), batch['target_ids'], deterministic=True, memory_mode='none')
        return jnp.argmax(logits, axis=-1), sim

    @jax.jit
    def eval_step_nn(params, mem_state, fact_store, write_idx, batch):
        h_eos_proj, _ = mdl.apply({'params': params}, batch['write_ids'], batch['write_mask'], True, jnp.ones(batch_size), deterministic=True, memory_mode='none', method=mdl.write_only)
        new_fact_store = jax.lax.dynamic_update_slice(fact_store, h_eos_proj, (write_idx, 0))
        logits, sim, _, _ = mdl.apply({'params': params, 'memory': mem_state}, batch['query_ids'], batch['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), batch['target_ids'], deterministic=True, memory_mode='nn', fact_store=new_fact_store)
        return jnp.argmax(logits, axis=-1), sim, new_fact_store
        
    @jax.jit
    def eval_step_bank(params, mem_state, batch):
        return None
    
    # Pre-extract all train batches into a single stacked dictionary for XLA scan
    all_train_batches = list(train_loader.iter_batches(shuffle=False))
    batched_train_data = {
        k: jnp.stack([b[k] for b in all_train_batches]) for k in all_train_batches[0].keys() if k not in ['valid_count', 'fact_str_ids', 'query_str_ids']
    }
    
    for seed in seeds:
        print(f"\n[Seed {seed}] Initializing Model Backbone...")
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        
        # Init base memory state (dummy) to allow parameter initialization
        dummy_mem = create_initial_memory(config)
        initial_vars = mdl.init(init_rng, dummy_input, dummy_input, jnp.ones(batch_size), jnp.ones(batch_size), jnp.ones(batch_size), dummy_target, method=mdl.init_all)
        initial_params = initial_vars['params']
        
        for name, mode in modes.items():
            print(f"  --- Training & Evaluating {name} (Seed {seed}) ---")
            
            # Reset mode_rng to ensure exactly identical dropout schedules across baselines
            mode_rng = jax.random.PRNGKey(seed)
            
            params = initial_params
            tx = optax.adamw(learning_rate=1e-3)
            opt_state = tx.init(params)
            
            # Persistent train episodic memory
            mem_state = initial_vars['memory']
            fact_store = jnp.zeros((config.memory_capacity, config.hidden_size))
            write_idx = 0
            
            for epoch in range(num_epochs):
                start_time = time.time()
                mode_rng, epoch_rng = jax.random.split(mode_rng)
                
                if mode == 'none':
                    params, opt_state, mem_state, losses = train_epoch_none(params, opt_state, mem_state, batched_train_data, epoch_rng)
                elif mode == 'nn':
                    params, opt_state, fact_store, write_idx, losses = train_epoch_nn(params, opt_state, mem_state, fact_store, write_idx, batched_train_data, epoch_rng)
                else:
                    params, opt_state, mem_state, losses = train_epoch_bank(params, opt_state, mem_state, batched_train_data, epoch_rng)
                
                # Block to measure time properly
                losses = jax.block_until_ready(losses)
                
                end_time = time.time()
                epoch_time = end_time - start_time
                steps = len(losses)
                step_time_ms = (epoch_time / max(steps, 1)) * 1000
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"    Epoch {epoch+1:3d} Loss: {np.mean(losses):.4f} | Time: {epoch_time:.2f}s | {step_time_ms:.2f} ms/step")
                
            # Evaluation
            ems, f1s, r1s, mrrs = [], [], [], []
            
            # Reset episode state for test set
            eval_mem_state = initial_vars['memory']
            eval_fact_store = jnp.zeros((config.memory_capacity, config.hidden_size))
            write_idx = 0
            
            fact_to_slot = {}
            slot_to_fact = {}
            
            for batch in loader.iter_batches(shuffle=False):
                mode_rng, step_rng = jax.random.split(mode_rng)
                jax_batch = {k: v for k, v in batch.items() if k not in ('fact_str_ids', 'query_str_ids')}
                
                if mode == 'none':
                    preds, sim = eval_step_none(params, eval_mem_state, jax_batch)
                elif mode == 'nn':
                    preds, sim, eval_fact_store = eval_step_nn(params, eval_mem_state, eval_fact_store, write_idx, jax_batch)
                    # For NN Memory, it writes sequentially
                    written_indices = jnp.arange(write_idx, write_idx + batch_size) % config.memory_capacity
                else:
                    written_indices, updated_mem = mdl.apply({'params': params, 'memory': eval_mem_state}, jax_batch['write_ids'], jax_batch['write_mask'], jnp.ones(batch_size), jnp.ones(batch_size), True, method=mdl.write_only, mutable=['memory'])
                    out, new_mem = mdl.apply({'params': params, 'memory': updated_mem['memory']}, jax_batch['query_ids'], jax_batch['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), jax_batch['target_ids'], deterministic=True, memory_mode='bank', mutable=['memory'])
                    logits, sim, _, _ = out
                    preds = jnp.argmax(logits, axis=-1)
                    eval_mem_state = new_mem['memory']
                
                
                # Slice out padded batch elements
                valid_count = batch['valid_count']
                preds = preds[:valid_count]
                targets = batch['target_ids'][:valid_count]
                batch_fact_ids = batch['fact_str_ids'][:valid_count]
                batch_query_ids = batch['query_str_ids'][:valid_count]
                
                em = exact_match(preds, targets, pad_id=loader.pad_id)
                f1 = batch_token_f1(preds, targets, pad_id=loader.pad_id)
                
                if mode != 'none':
                    # Update true ground truth mappings
                    for i in range(valid_count):
                        fid = batch_fact_ids[i]
                        slot = int(written_indices[i])
                        
                        old_fact = slot_to_fact.get(slot)
                        if old_fact is not None and old_fact != fid:
                            fact_to_slot[old_fact] = None # Evicted
                            
                        slot_to_fact[slot] = fid
                        fact_to_slot[fid] = slot
                    
                    # Evaluate retrieval
                    for i in range(valid_count):
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
                write_idx = (write_idx + batch_size) % config.memory_capacity
                
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
