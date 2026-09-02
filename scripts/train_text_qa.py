import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from tqdm import tqdm

from models.transformer_qa_model import TransformerQAModel
from models.tiny_memory_bank import TinyMemoryConfig
from dataset.text_dataset_loader import TextDataLoader

def cross_entropy_loss(logits, targets, pad_id, vocab_size):
    one_hot = jax.nn.one_hot(targets, vocab_size)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    mask = (targets != pad_id).astype(jnp.float32)
    return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

def main():
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
    train_csv = os.path.join(dataset_dir, "train.csv")
    val_csv = os.path.join(dataset_dir, "val.csv")
    tokenizer_path = os.path.join(dataset_dir, "tokenizer.json")
    
    # 1. Hyperparameters
    vocab_size = 2000
    batch_size = 32
    max_input_len = 16
    max_target_len = 16
    
    # 2. Setup Data
    train_loader = TextDataLoader(train_csv, tokenizer_path, batch_size, max_input_len, max_target_len)
    val_loader = TextDataLoader(train_csv, tokenizer_path, batch_size, max_input_len, max_target_len)
    pad_id = train_loader.pad_id
    
    # Disable thresholds during training to prevent cold-start bypass and enable gradients
    config = TinyMemoryConfig(
        memory_capacity=128,
        memory_dim=32,
        hidden_size=32,
        memory_threshold=0.0,
        memory_read_threshold=0.0,
        memory_write_threshold=0.7,
        memory_top_k=8
    )
    model = TransformerQAModel(config=config, vocab_size=vocab_size, embed_dim=32, num_layers=1, num_heads=2, ff_dim=64, max_target_len=max_target_len, dropout_rate=0.0)
    num_epochs = 10
    
    # Cosine decay schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4,
        peak_value=2e-3,
        warmup_steps=100,
        decay_steps=num_epochs * train_loader.num_batches,
        end_value=1e-5
    )
    
    # Optimizer dengan Weight Decay dan Gradient Clipping untuk mencegah NaN
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)
    )
    
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    dummy_input = jnp.ones((batch_size, max_input_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((batch_size, max_input_len), dtype=jnp.int32)
    dummy_target = jnp.ones((batch_size, max_target_len), dtype=jnp.int32)
    dummy_p = jnp.ones((batch_size,), dtype=jnp.float32)
    
    print("Inisialisasi Model Parameter...")
    variables = model.init(init_rng, dummy_input, dummy_mask, dummy_p, dummy_p, dummy_p, dummy_target, method=model.init_all)
    
    state = {'params': variables['params']}
    opt_state = tx.init(state['params'])
    blank_memory = variables['memory']
    
    @jax.jit
    def train_step(params, opt_state, batch, step_rng):
        write_rng, dropout_rng = jax.random.split(step_rng)
        
        def loss_fn(params):
            vars = {'params': params, 'memory': blank_memory}
            is_eos = jnp.ones((batch_size,))
            write_p = jnp.ones((batch_size,))
            
            # Forward pass (Training)
            (_, h_eos), updated_memory = model.apply(vars, batch['write_ids'], batch['write_mask'], is_eos, write_p, 
                                            deterministic=False, method=model.write_only, mutable=['memory'], rngs={'dropout': dropout_rng})
                                            
            new_vars = {'params': params, 'memory': updated_memory['memory']}
            read_p = jnp.ones((batch_size,))
            write_p_zero = jnp.zeros((batch_size,))
            
            (logits, sim, query_h_eos, h_fused), _ = model.apply(new_vars, batch['query_ids'], batch['query_mask'], 
                                    read_p, write_p_zero, batch['target_ids'], deterministic=False, 
                                    mutable=['memory'], rngs={'dropout': dropout_rng})
            
            # Auto-regressive loss (Teacher Forcing)
            mask = (batch['target_ids'] != pad_id)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['target_ids'])
            loss = (loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
            
            # Contrastive Loss on Memory Retrieval
            labels = jnp.arange(batch_size)
            target_sim = jax.nn.one_hot(labels, sim.shape[-1])
            aux_loss = jnp.mean((sim - target_sim) ** 2)
            
            # Orthogonal Contrastive Loss for Encoder Representations (h_eos)
            h_eos_pooled = jnp.mean(h_eos, axis=1)
            h_norm = h_eos_pooled / jnp.sqrt(jnp.sum(h_eos_pooled**2, axis=-1, keepdims=True) + 1e-8)
            sim_matrix = jnp.matmul(h_norm, h_norm.T) # shape: (batch_size, batch_size)
            off_diagonal_mask = 1.0 - jnp.eye(batch_size)
            # Penalty if similarity is greater than 0 for different facts
            contrast_loss = jnp.mean((sim_matrix * off_diagonal_mask) ** 2)
            
            # Query-Fact Contrastive Loss (InfoNCE): Align h_query (Query GRU) dengan h_fact (Fact GRU)
            h_fact_eos = model.apply({'params': params, 'memory': blank_memory},
                                     batch['write_ids'], batch['write_mask'],
                                     deterministic=False, method=model.encode_fact,
                                     rngs={'dropout': dropout_rng})
            
            h_fact_pooled = jnp.mean(h_fact_eos, axis=1)
            h_q_norm = h_norm
            h_f_norm = h_fact_pooled / jnp.sqrt(jnp.sum(h_fact_pooled**2, axis=-1, keepdims=True) + 1e-8)
            
            # Cosine similarity matrix antar query_i dan fact_j dalam batch (scaled by temperature 0.1)
            qf_sim_matrix = jnp.matmul(h_q_norm, h_f_norm.T) / 0.1
            qf_labels = jnp.arange(batch_size)
            qf_contrastive_loss = optax.softmax_cross_entropy_with_integer_labels(qf_sim_matrix, qf_labels).mean()

            # Retrieval Supervision Loss: h_fused harus semirip mungkin dengan h_fact_eos
            h_fused_pooled = jnp.mean(h_fused, axis=1)
            h_fused_norm = h_fused_pooled / jnp.sqrt(jnp.sum(h_fused_pooled**2, axis=-1, keepdims=True) + 1e-8)
            retrieval_sim = jnp.sum(h_fused_norm * h_f_norm, axis=-1)
            retrieval_loss = 1.0 - jnp.mean(retrieval_sim)
            
            # Oracle Decode Loss: latih decoder langsung dari h_fact_eos (oracle supervision)
            oracle_logits = model.apply(
                {'params': params, 'memory': blank_memory},
                batch['write_ids'], batch['write_mask'],
                batch['query_ids'], batch['query_mask'],
                batch['target_ids'],
                deterministic=False, method=model.decode_oracle,
                rngs={'dropout': dropout_rng}
            )
            oracle_loss = optax.softmax_cross_entropy_with_integer_labels(oracle_logits, batch['target_ids'])
            oracle_loss = (oracle_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
            
            total_loss = loss + 10.0 * aux_loss + 2.0 * contrast_loss + 5.0 * retrieval_loss + 3.0 * oracle_loss + 5.0 * qf_contrastive_loss
            
            # Calculate accuracy
            preds = jnp.argmax(logits, axis=-1)
            correct = jnp.sum((preds == batch['target_ids']) * mask)
            total = jnp.maximum(jnp.sum(mask), 1.0)
            acc = correct / total
            
            return total_loss, acc
            
        (total_loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params) # pass params for weight decay!
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, total_loss

    @jax.jit
    def eval_step(params, batch):
        vars = {'params': params, 'memory': blank_memory}
        is_eos = jnp.ones((batch_size,))
        write_p = jnp.ones((batch_size,))
        
        # Tulis fakta ke memori
        (_, h_eos), updated_memory = model.apply(vars, batch['write_ids'], batch['write_mask'], is_eos, write_p, 
                                        deterministic=True, method=model.write_only, mutable=['memory'])
        
        new_vars = {'params': params, 'memory': updated_memory['memory']}
        read_p = jnp.ones((batch_size,))
        write_p_zero = jnp.zeros((batch_size,))
        
        # --- Hitung loss (pakai teacher forcing hanya untuk loss, bukan untuk acc) ---
        (logits, sim, _, _), _ = model.apply(new_vars, batch['query_ids'], batch['query_mask'], 
                                read_p, write_p_zero, batch['target_ids'], deterministic=True, mutable=['memory'])
        loss = cross_entropy_loss(logits, batch['target_ids'], pad_id, vocab_size)
        
        # --- Hitung accuracy menggunakan GREEDY DECODE dengan EXACT MATCH ---
        preds, _ = model.apply(new_vars, batch['query_ids'], batch['query_mask'],
                            read_p, write_p_zero, max_target_len, 2, pad_id, 3,
                            deterministic=True, method=model.greedy_decode, mutable=['memory'])
        
        # Exact Match: semua token non-PAD harus sama persis
        mask = (batch['target_ids'] != pad_id)
        token_equals = (preds == batch['target_ids']) | (~mask)
        sequence_exact_match = jnp.all(token_equals, axis=-1)  # (batch_size,)
        acc = jnp.mean(sequence_exact_match.astype(jnp.float32))
        
        return loss, acc

    print(f"Mulai training NLP (GRU Autoregressive) selama {num_epochs} epoch...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        train_losses = []
        num_batches_yielded = 0
        num_batches_processed = 0
        
        for batch in train_loader.iter_batches(shuffle=True):
            num_batches_yielded += 1
            if len(batch['write_ids']) < batch_size:
                continue
            num_batches_processed += 1
                
            rng, step_rng = jax.random.split(rng)
            state['params'], opt_state, loss = train_step(state['params'], opt_state, batch, step_rng)
            train_losses.append(loss)
            
            if np.isnan(loss):
                print(f"NaN generated BY MODEL at Epoch {epoch}, Batch {len(train_losses)}!", flush=True)
                raise ValueError(f"NaN detected at Epoch {epoch}, Batch {len(train_losses)}!")
        
        print(f"DEBUG: Epoch {epoch} yielded {num_batches_yielded} batches, processed {num_batches_processed} batches.")
            
        val_losses = []
        val_accs = []
        for batch in val_loader.iter_batches(shuffle=False):
            if len(batch['write_ids']) < batch_size:
                continue
                
            v_loss, v_acc = eval_step(state['params'], batch)
            val_losses.append(v_loss)
            val_accs.append(v_acc)
            
        avg_t_loss = np.mean(train_losses)
        avg_v_loss = np.mean(val_losses)
        avg_v_acc = np.mean(val_accs)
        
        mark = "✓ best" if avg_v_loss < best_val_loss else ""
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            
        print(f"Epoch {epoch:03d} | Train Loss: {avg_t_loss:.4f} | Val Loss: {avg_v_loss:.4f} | Val Acc: {avg_v_acc*100:.1f}% {mark}")

    print("=" * 60)
    print("MENGUJI MODEL (GREEDY DECODING) PADA DATA TEST")
    print("=" * 60)
    
    try:
        from flax import serialization
        ckpt_path = os.path.abspath(os.path.join(dataset_dir, "..", "results", "model.msgpack"))
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        with open(ckpt_path, "wb") as f:
            f.write(serialization.to_bytes(state['params']))
        print(f"Model tersimpan di {ckpt_path}")
    except Exception as e:
        print(f"Gagal menyimpan model: {e}")
        
    test_csv = os.path.join(dataset_dir, "test.csv")
    test_loader = TextDataLoader(test_csv, tokenizer_path, batch_size, max_input_len, max_target_len)
    
    @jax.jit
    def generate_step(params, batch):
        vars = {'params': params, 'memory': blank_memory}
        is_eos = jnp.ones((batch_size,))
        write_p = jnp.ones((batch_size,))
        
        _, updated_memory = model.apply(vars, batch['write_ids'], batch['write_mask'], is_eos, write_p, 
                                        deterministic=True, method=model.write_only, mutable=['memory'])
        
        new_vars = {'params': params, 'memory': updated_memory['memory']}
        read_p = jnp.ones((batch_size,))
        write_p_zero = jnp.zeros((batch_size,))
        
        preds, _ = model.apply(new_vars, batch['query_ids'], batch['query_mask'], 
                            read_p, write_p_zero, max_target_len, 2, pad_id, 3, # bos_id, pad_id, eos_id
                            deterministic=True, method=model.greedy_decode, mutable=['memory'])
        return preds
        
    for batch in test_loader.iter_batches(shuffle=True):
        if len(batch['write_ids']) < batch_size:
            continue
            
        preds = generate_step(state['params'], batch)
        
        for i in range(5):
            print(f"\n--- Sampel {i+1} ---")
            q_str = test_loader.tokenizer.decode(batch['query_ids'][i].tolist(), skip_special_tokens=True)
            t_str = test_loader.tokenizer.decode(batch['target_ids'][i].tolist(), skip_special_tokens=True)
            p_str = test_loader.tokenizer.decode(preds[i].tolist(), skip_special_tokens=True)
            
            print(f"Query   : {q_str}")
            print(f"Target  : {t_str}")
            print(f"Prediksi: {p_str}")
            
        break

if __name__ == "__main__":
    main()
