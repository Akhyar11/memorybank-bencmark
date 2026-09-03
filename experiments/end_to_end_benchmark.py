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

def cross_entropy_loss(logits, targets, pad_id, vocab_size=2000):
    one_hot = jax.nn.one_hot(targets, vocab_size)
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(one_hot * log_prob, axis=-1)
    mask = (targets != pad_id).astype(jnp.float32)
    return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

def run_benchmark():
    print("===========================================")
    print("      END-TO-END MEMORY BENCHMARK          ")
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
    
    config = TinyMemoryConfig(memory_capacity=128, memory_dim=32, hidden_size=32)
    
    # We use ONE single backbone
    mdl = TransformerQAModel(config=config, vocab_size=2000, embed_dim=32, num_layers=1, num_heads=2, pad_id=loader.pad_id, bos_id=loader.bos_id, eos_id=loader.eos_id)
    
    dummy_input = jnp.ones((batch_size, 32), dtype=jnp.int32)
    dummy_target = jnp.ones((batch_size, 16), dtype=jnp.int32)
    
    modes = {"No Memory": "none", "NN Memory": "nn", "Memory Bank": "bank"}
    num_epochs = 1000
    seeds = [42, 43, 44]
    
    results = {name: collections.defaultdict(list) for name in modes.keys()}

    @jax.jit
    def train_step_none(params, mem_state, batch, rng):
        def loss_fn(p):
            logits, _, _, _ = mdl.apply({'params': p, 'memory': mem_state}, batch['query_ids'], batch['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), batch['target_ids'], deterministic=False, memory_mode='none', rngs={'dropout': rng})
            return cross_entropy_loss(logits, batch['target_ids'], loader.pad_id)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        return grads, loss

    @jax.jit
    def train_step_nn(params, mem_state, fact_store, write_idx, batch, rng):
        def loss_fn(p):
            h_eos_proj = mdl.apply({'params': p}, batch['write_ids'], batch['write_mask'], True, jnp.ones(batch_size), deterministic=False, memory_mode='none', method=mdl.write_only, rngs={'dropout': rng})
            new_fact_store = jax.lax.dynamic_update_slice(fact_store, h_eos_proj, (write_idx, 0))
            logits, _, _, _ = mdl.apply({'params': p, 'memory': mem_state}, batch['query_ids'], batch['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), batch['target_ids'], deterministic=False, memory_mode='nn', fact_store=new_fact_store, rngs={'dropout': rng})
            return cross_entropy_loss(logits, batch['target_ids'], loader.pad_id), new_fact_store
        (loss, new_fs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return grads, loss, new_fs

    @jax.jit
    def train_step_bank(params, mem_state, batch, rng):
        def loss_fn(p):
            _, updated_mem = mdl.apply({'params': p, 'memory': mem_state}, batch['write_ids'], batch['write_mask'], jnp.ones(batch_size), jnp.ones(batch_size), True, method=mdl.write_only, mutable=['memory'], rngs={'dropout': rng})
            logits, _, _, _ = mdl.apply({'params': p, 'memory': updated_mem['memory']}, batch['query_ids'], batch['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), batch['target_ids'], deterministic=False, memory_mode='bank', rngs={'dropout': rng}, mutable=['memory'])[0]
            return cross_entropy_loss(logits, batch['target_ids'], loader.pad_id), updated_mem['memory']
        (loss, new_mem), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return grads, loss, new_mem
        
    @jax.jit
    def eval_step_none(params, mem_state, batch):
        logits, sim, _, _ = mdl.apply({'params': params, 'memory': mem_state}, batch['query_ids'], batch['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), batch['target_ids'], deterministic=True, memory_mode='none')
        return jnp.argmax(logits, axis=-1), sim

    @jax.jit
    def eval_step_nn(params, mem_state, fact_store, write_idx, batch):
        h_eos_proj = mdl.apply({'params': params}, batch['write_ids'], batch['write_mask'], True, jnp.ones(batch_size), deterministic=True, memory_mode='none', method=mdl.write_only)
        new_fact_store = jax.lax.dynamic_update_slice(fact_store, h_eos_proj, (write_idx, 0))
        logits, sim, _, _ = mdl.apply({'params': params, 'memory': mem_state}, batch['query_ids'], batch['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), batch['target_ids'], deterministic=True, memory_mode='nn', fact_store=new_fact_store)
        return jnp.argmax(logits, axis=-1), sim, new_fact_store
        
    @jax.jit
    def eval_step_bank(params, mem_state, batch):
        _, updated_mem = mdl.apply({'params': params, 'memory': mem_state}, batch['write_ids'], batch['write_mask'], jnp.ones(batch_size), jnp.ones(batch_size), True, method=mdl.write_only, mutable=['memory'])
        out, new_mem = mdl.apply({'params': params, 'memory': updated_mem['memory']}, batch['query_ids'], batch['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), batch['target_ids'], deterministic=True, memory_mode='bank', mutable=['memory'])
        logits, sim, _, _ = out
        return jnp.argmax(logits, axis=-1), sim, new_mem['memory']
    
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
            params = initial_params
            tx = optax.adamw(learning_rate=1e-3)
            opt_state = tx.init(params)
            
            # Persistent train episodic memory
            mem_state = create_initial_memory(config)
            fact_store = jnp.zeros((config.memory_capacity, config.hidden_size))
            
            for epoch in range(num_epochs):
                losses = []
                write_idx = 0
                for batch in train_loader.iter_batches(shuffle=False): # Sequential for episodic integrity!
                    rng, step_rng = jax.random.split(rng)
                    if mode == 'none':
                        grads, loss = train_step_none(params, mem_state, batch, step_rng)
                    elif mode == 'nn':
                        grads, loss, fact_store = train_step_nn(params, mem_state, fact_store, write_idx, batch, step_rng)
                    else:
                        grads, loss, mem_state = train_step_bank(params, mem_state, batch, step_rng)
                    
                    updates, opt_state = tx.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    losses.append(loss)
                    write_idx = (write_idx + batch_size) % config.memory_capacity
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"    Epoch {epoch+1:3d} Loss: {np.mean(losses):.4f}")
                
            # Evaluation
            ems, f1s, r1s, mrrs = [], [], [], []
            
            # Reset episode state for test set
            eval_mem_state = create_initial_memory(config)
            eval_fact_store = jnp.zeros((config.memory_capacity, config.hidden_size))
            write_idx = 0
            
            for batch in loader.iter_batches(shuffle=False):
                if mode == 'none':
                    preds, sim = eval_step_none(params, eval_mem_state, batch)
                elif mode == 'nn':
                    preds, sim, eval_fact_store = eval_step_nn(params, eval_mem_state, eval_fact_store, write_idx, batch)
                else:
                    preds, sim, eval_mem_state = eval_step_bank(params, eval_mem_state, batch)
                
                em = exact_match(preds, batch['target_ids'], pad_id=loader.pad_id)
                f1 = batch_token_f1(preds, batch['target_ids'], pad_id=loader.pad_id)
                
                # Retrieval Metrics (Ground truth is write_idx to write_idx + batch_size - 1)
                gt_indices = jnp.arange(write_idx, write_idx + batch_size)
                
                if mode != 'none':
                    for i in range(batch_size):
                        gt_idx = int(write_idx + i)
                        r1 = recall_at_k(np.array(sim[i]), gt_idx, k_values=[1])[1]
                        mrr = mean_reciprocal_rank(np.array(sim[i]), gt_idx)
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
