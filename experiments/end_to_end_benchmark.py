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
import flax.core
import flax.linen as nn
import numpy as np

from models.transformer_qa_model import TransformerQAModel
from models.tiny_memory_bank import TinyMemoryConfig, STATE_ACTIVE
from dataset.text_dataset_loader import TextDataLoader
from evaluation.metrics import exact_match, batch_token_f1, recall_at_k, mean_reciprocal_rank

class NoMemoryQAModel(TransformerQAModel):
    def __call__(self, input_ids, mask, read_prob, write_prob, target_ids, deterministic=False):
        h_eos = self.encode_query(input_ids, mask, deterministic=deterministic)
        h_eos_proj = self.memory_proj_in(h_eos)
        h_fused_proj = jnp.zeros((h_eos.shape[0], self.embed_dim))
        context = h_fused_proj[:, None, :]
        logits = self._decode_from_context(context, target_ids, deterministic=deterministic)
        return logits, jnp.zeros((h_eos.shape[0], self.config.memory_capacity)), h_eos_proj, jnp.zeros_like(h_eos_proj)

class NNMemoryQAModel(TransformerQAModel):
    def __call__(self, input_ids, mask, read_prob, write_prob, target_ids, deterministic=False):
        h_eos = self.encode_query(input_ids, mask, deterministic=deterministic)
        h_eos_proj = self.memory_proj_in(h_eos)
        
        # Access raw memory from bank
        # We need to compute attention manually
        keys = self.bank.mem_keys.value
        vals = self.bank.mem_vals.value
        state = self.bank.mem_state.value
        q = self.bank.q_proj(h_eos_proj)
        q_n = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
        k_n = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        sim = jnp.dot(q_n, k_n.T)
        active_mask = (state != 0)[None, :]
        sim = jnp.where(active_mask, sim, -1e9)
        attn = jax.nn.softmax(sim, axis=-1)
        m = jnp.dot(attn, vals)
        h_fused = self.bank.fusion_proj(jnp.concatenate([h_eos_proj, m], axis=-1))
        h_fused_proj = self.memory_proj_out(h_fused)
        
        context = h_fused_proj[:, None, :]
        logits = self._decode_from_context(context, target_ids, deterministic=deterministic)
        return logits, jnp.zeros((h_eos.shape[0], self.config.memory_capacity)), h_eos_proj, h_fused

import optax

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
    train_loader = TextDataLoader(train_csv, tokenizer_path, batch_size, 32, 16, max_samples=128)
    # Test loader
    loader = TextDataLoader(test_csv, tokenizer_path, batch_size, 32, 16, max_samples=32)
    
    config = TinyMemoryConfig(memory_capacity=128, memory_dim=32, hidden_size=32)
    
    models = {
        "No Memory": NoMemoryQAModel(config=config, vocab_size=2000, embed_dim=32, num_layers=1, num_heads=2, pad_id=loader.pad_id, bos_id=loader.bos_id, eos_id=loader.eos_id),
        "NN Memory": NNMemoryQAModel(config=config, vocab_size=2000, embed_dim=32, num_layers=1, num_heads=2, pad_id=loader.pad_id, bos_id=loader.bos_id, eos_id=loader.eos_id),
        "Memory Bank": TransformerQAModel(config=config, vocab_size=2000, embed_dim=32, num_layers=1, num_heads=2, pad_id=loader.pad_id, bos_id=loader.bos_id, eos_id=loader.eos_id)
    }
    
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((batch_size, 32), dtype=jnp.int32)
    dummy_target = jnp.ones((batch_size, 16), dtype=jnp.int32)
    
    # Simple cross entropy loss
    def cross_entropy_loss(logits, targets, pad_id, vocab_size=2000):
        one_hot = jax.nn.one_hot(targets, vocab_size)
        log_prob = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(one_hot * log_prob, axis=-1)
        mask = (targets != pad_id).astype(jnp.float32)
        return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    
    for name, mdl in models.items():
        print(f"\n--- Training & Evaluating {name} ---")
        rng, init_rng = jax.random.split(rng)
        vars = mdl.init(init_rng, dummy_input, dummy_input, jnp.ones(batch_size), jnp.ones(batch_size), jnp.ones(batch_size), dummy_target, method=mdl.init_all)
        params = vars['params']
        
        tx = optax.adamw(learning_rate=1e-3)
        opt_state = tx.init(params)
        
        @jax.jit
        def train_step(params, batch, rng):
            def loss_fn(p):
                mem = {
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
                _, new_mem = mdl.apply({'params': p, 'memory': mem}, batch['write_ids'], batch['write_mask'], jnp.ones(batch_size), jnp.ones(batch_size), True, method=mdl.write_only, mutable=['memory'], rngs={'dropout': rng})
                logits, _, _, _ = mdl.apply({'params': p, 'memory': new_mem['memory']}, batch['query_ids'], batch['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), batch['target_ids'], deterministic=False, rngs={'dropout': rng}, mutable=['memory'])[0]
                return cross_entropy_loss(logits, batch['target_ids'], loader.pad_id)
            
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state_new = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state_new, loss
        
        print("Training for 50 epochs...")
        for epoch in range(50):
            losses = []
            for batch in train_loader.iter_batches(shuffle=True):
                rng, step_rng = jax.random.split(rng)
                params, opt_state, loss = train_step(params, batch, step_rng)
                losses.append(loss)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} Loss: {np.mean(losses):.4f}")
            
        vars = {'params': params}
        
        ems, f1s = [], []
        for batch in loader.iter_batches(shuffle=False):
            # Write phase
            _, updated_mem = mdl.apply(vars, batch['write_ids'], batch['write_mask'], jnp.ones(batch_size), jnp.ones(batch_size), True, method=mdl.write_only, mutable=['memory'])
            
            # Read phase (greedy decode for test)
            # Uniform evaluation using teacher-forced logits for all models
            logits, _, _, _ = mdl.apply({'params': vars['params'], 'memory': updated_mem['memory']}, batch['query_ids'], batch['query_mask'], jnp.ones(batch_size), jnp.zeros(batch_size), batch['target_ids'], deterministic=True, mutable=['memory'])[0]
            preds = jnp.argmax(logits, axis=-1)
            em = exact_match(preds, batch['target_ids'], pad_id=loader.pad_id)
            f1 = batch_token_f1(preds, batch['target_ids'], pad_id=loader.pad_id)
            
            ems.append(em)
            f1s.append(f1)
            
        print(f"  Exact Match: {np.mean(ems)*100:.2f}%")
        print(f"  Token F1:    {np.mean(f1s)*100:.2f}%")

if __name__ == '__main__':
    run_benchmark()
