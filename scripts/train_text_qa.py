"""
train_text_qa.py – Fixed QA training script.

Fixes applied:
- BUG-P0-019: val_loader now uses val_csv (not train_csv)
- BUG-P0-021: Oracle loss REMOVED from total_loss; kept as separate diagnostic metric
- BUG-P0-022: Test loop evaluates ENTIRE test set (no break)
- BUG-P0-023: TextDataLoader uses max_samples=None (no silent truncation)
- BUG-P0-024: blank_memory refreshed properly per training step
- BUG-P1-004: Comment corrected – free-running autoregressive, not teacher forcing
- BUG-P1-008: Loss components logged separately; QA loss is primary objective
- Loss: decoder outputs logits → cross_entropy_loss (not log(probs))
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
import yaml

from models.transformer_qa_model import TransformerQAModel
from models.tiny_memory_bank import TinyMemoryConfig, STATE_EXPIRED
from dataset.text_dataset_loader import TextDataLoader


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def cross_entropy_loss(logits, targets, pad_id, vocab_size):
    """
    Standard cross-entropy from logits.
    logits: (batch, seq, vocab_size) – raw logits (NOT softmax'd)
    targets: (batch, seq)
    """
    one_hot  = jax.nn.one_hot(targets, vocab_size)
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    loss     = -jnp.sum(one_hot * log_prob, axis=-1)
    mask     = (targets != pad_id).astype(jnp.float32)
    return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def make_blank_memory(config):
    """Create a truly empty memory state (all slots EXPIRED)."""
    cap = config.memory_capacity
    dim = config.memory_dim
    return {
        'keys':         jnp.zeros((cap, dim),  dtype=jnp.float32),
        'vals':         jnp.zeros((cap, dim),  dtype=jnp.float32),
        'importance':   jnp.zeros((cap,),      dtype=jnp.float32),
        'confidence':   jnp.zeros((cap,),      dtype=jnp.float32),
        'created_at':   jnp.zeros((cap,),      dtype=jnp.int32),
        'last_access':  jnp.zeros((cap,),      dtype=jnp.int32),
        'access_count': jnp.zeros((cap,),      dtype=jnp.int32),
        'state':        jnp.full((cap,), STATE_EXPIRED, dtype=jnp.int32),
        'global_step':  jnp.zeros((),          dtype=jnp.int32),
    }


def main():
    dataset_dir    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    train_csv      = os.path.join(dataset_dir, 'train.csv')
    val_csv        = os.path.join(dataset_dir, 'val.csv')   # FIX BUG-P0-019
    test_csv       = os.path.join(dataset_dir, 'test.csv')
    tokenizer_path = os.path.join(dataset_dir, 'tokenizer.json')

    # Verify distinct dataset files
    assert train_csv != val_csv,  "train and val CSV must be different files!"
    assert train_csv != test_csv, "train and test CSV must be different files!"

    # ---------------------------------------------------------------------------
    # Load Hyperparameters from configs/benchmark.yaml
    # ---------------------------------------------------------------------------
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'benchmark.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    vocab_size     = cfg.get('vocab_size', 2000)
    batch_size     = cfg.get('batch_size', 32)
    max_input_len  = cfg.get('max_input_len', 32)
    max_target_len = cfg.get('max_target_len', 16)
    num_epochs     = cfg.get('num_epochs', 10)
    seed           = cfg.get('seed', 42)
    
    embed_dim      = cfg.get('embed_dim', 32)
    num_layers     = cfg.get('num_layers', 1)
    num_heads      = cfg.get('num_heads', 2)
    ff_dim         = cfg.get('ff_dim', 64)
    dropout_rate   = cfg.get('dropout_rate', 0.0)

    # ---------------------------------------------------------------------------
    # Data loaders – max_samples=None loads full dataset
    # ---------------------------------------------------------------------------
    train_loader = TextDataLoader(train_csv, tokenizer_path, batch_size,
                                  max_input_len, max_target_len, max_samples=None)
    val_loader   = TextDataLoader(val_csv,   tokenizer_path, batch_size,
                                  max_input_len, max_target_len, max_samples=None)
    pad_id = train_loader.pad_id

    # ---------------------------------------------------------------------------
    # Model config
    # ---------------------------------------------------------------------------
    config = TinyMemoryConfig(
        memory_capacity      = cfg.get('memory_capacity', 128),
        memory_dim           = cfg.get('memory_dim', 32),
        hidden_size          = cfg.get('hidden_size', 32),
        memory_threshold     = 0.0,  # disable threshold during training (cold-start)
        memory_read_threshold= 0.0,
        memory_write_threshold= 0.0,
        memory_top_k         = cfg.get('memory_top_k', 8),
    )

    model = TransformerQAModel(
        config        = config,
        vocab_size    = vocab_size,
        embed_dim     = embed_dim,
        num_layers    = num_layers,
        num_heads     = num_heads,
        ff_dim        = ff_dim,
        max_target_len= max_target_len,
        dropout_rate  = dropout_rate,
    )

    # ---------------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------------
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value   = 1e-4,
        peak_value   = 2e-3,
        warmup_steps = 100,
        decay_steps  = num_epochs * train_loader.num_batches,
        end_value    = 1e-5,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4),
    )

    # ---------------------------------------------------------------------------
    # Initialise model
    # ---------------------------------------------------------------------------
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    dummy_input  = jnp.ones((batch_size, max_input_len),  dtype=jnp.int32)
    dummy_mask   = jnp.ones((batch_size, max_input_len),  dtype=jnp.int32)
    dummy_target = jnp.ones((batch_size, max_target_len), dtype=jnp.int32)
    dummy_p      = jnp.ones((batch_size,),                dtype=jnp.float32)

    print("Initialising model parameters...")
    variables = model.init(
        init_rng, dummy_input, dummy_mask, dummy_p, dummy_p, dummy_p, dummy_target,
        method=model.init_all
    )

    params    = variables['params']
    opt_state = tx.init(params)

    # blank_memory is recreated per training step (episodic: memory resets each batch)
    blank_memory = make_blank_memory(config)

    # ---------------------------------------------------------------------------
    # Training step
    # ---------------------------------------------------------------------------
    @jax.jit
    def train_step(params, opt_state, batch, step_rng):
        dropout_rng, _ = jax.random.split(step_rng)

        def loss_fn(params):
            # Fresh memory per batch (episodic mode)
            mem = make_blank_memory(config)

            is_eos  = jnp.ones((batch_size,))
            write_p = jnp.ones((batch_size,))

            # --- WRITE phase ---
            h_eos_fact, updated_mem = model.apply(
                {'params': params, 'memory': mem},
                batch['write_ids'], batch['write_mask'],
                is_eos, write_p, True,
                method=model.write_only,
                mutable=['memory'],
                rngs={'dropout': dropout_rng},
            )

            # --- READ + FUSE + DECODE phase ---
            read_p    = jnp.ones((batch_size,))
            write_off = jnp.zeros((batch_size,))

            (logits, sim, h_eos_q, h_fused), _ = model.apply(
                {'params': params, 'memory': updated_mem['memory']},
                batch['query_ids'], batch['query_mask'],
                read_p, write_off, batch['target_ids'],
                deterministic=False,
                mutable=['memory'],
                rngs={'dropout': dropout_rng},
            )

            # --- Primary QA loss (logits → cross-entropy) ---
            # logits: (batch, seq, vocab) – raw logits from decoder
            qa_loss = cross_entropy_loss(logits, batch['target_ids'], pad_id, vocab_size)

            # --- Auxiliary: Query-Fact InfoNCE contrastive (optional, small weight) ---
            h_q_norm = h_eos_q / (jnp.linalg.norm(h_eos_q, axis=-1, keepdims=True) + 1e-8)

            h_f_norm_vals, _ = model.apply(
                {'params': params, 'memory': make_blank_memory(config)},
                batch['write_ids'], batch['write_mask'], True,
                method=model.encode_fact,
                mutable=['memory'],
                rngs={'dropout': dropout_rng},
            )
            h_f_vals = h_f_norm_vals
            h_f_norm = h_f_vals / (jnp.linalg.norm(h_f_vals, axis=-1, keepdims=True) + 1e-8)

            qf_sim    = jnp.matmul(h_q_norm, h_f_norm.T) / 0.1
            qf_labels = jnp.arange(batch_size)
            qf_loss   = optax.softmax_cross_entropy_with_integer_labels(qf_sim, qf_labels).mean()

            # --- Retrieval alignment: h_fused should be similar to h_fact ---
            h_fused_norm = h_fused / (jnp.linalg.norm(h_fused, axis=-1, keepdims=True) + 1e-8)
            retrieval_sim  = jnp.sum(h_fused_norm * h_f_norm, axis=-1)
            retrieval_loss = 1.0 - jnp.mean(retrieval_sim)

            # --- DIAGNOSTIC ONLY: oracle loss (not added to total) ---
            oracle_logits, _ = model.apply(
                {'params': params, 'memory': make_blank_memory(config)},
                batch['write_ids'], batch['write_mask'],
                batch['query_ids'], batch['query_mask'],
                batch['target_ids'], True,
                method=model.decode_oracle,
                mutable=['memory'],
                rngs={'dropout': dropout_rng},
            )
            oracle_loss = cross_entropy_loss(oracle_logits, batch['target_ids'], pad_id, vocab_size)

            # --- Total loss: QA loss primary; small aux weights ---
            total_loss = qa_loss + 0.5 * qf_loss + 0.5 * retrieval_loss

            # Token accuracy (training signal check)
            preds   = jnp.argmax(logits, axis=-1)
            mask    = (batch['target_ids'] != pad_id)
            correct = jnp.sum((preds == batch['target_ids']) * mask)
            acc     = correct / jnp.maximum(jnp.sum(mask), 1.0)

            return total_loss, (qa_loss, qf_loss, retrieval_loss, oracle_loss, acc)

        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, total_loss, aux

    # ---------------------------------------------------------------------------
    # Eval step
    # ---------------------------------------------------------------------------
    @jax.jit
    def eval_step(params, batch):
        mem     = make_blank_memory(config)
        is_eos  = jnp.ones((batch_size,))
        write_p = jnp.ones((batch_size,))

        _, updated_mem = model.apply(
            {'params': params, 'memory': mem},
            batch['write_ids'], batch['write_mask'],
            is_eos, write_p, True,
            method=model.write_only,
            mutable=['memory'],
        )

        read_p    = jnp.ones((batch_size,))
        write_off = jnp.zeros((batch_size,))

        (logits, _, _, _), _ = model.apply(
            {'params': params, 'memory': updated_mem['memory']},
            batch['query_ids'], batch['query_mask'],
            read_p, write_off, batch['target_ids'],
            deterministic=True,
            mutable=['memory'],
        )
        loss = cross_entropy_loss(logits, batch['target_ids'], pad_id, vocab_size)

        preds = model.apply(
            {'params': params, 'memory': updated_mem['memory']},
            batch['query_ids'], batch['query_mask'],
            read_p, write_off, max_target_len, 2, pad_id, 3,
            deterministic=True,
            method=model.greedy_decode,
            mutable=['memory'],
        )[0]

        mask        = (batch['target_ids'] != pad_id)
        tok_eq      = (preds == batch['target_ids']) | (~mask)
        exact_match = jnp.mean(jnp.all(tok_eq, axis=-1).astype(jnp.float32))

        return loss, exact_match

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    print(f"Starting training for {num_epochs} epochs...")
    print("-" * 70)

    best_val_loss    = float('inf')
    best_params      = params
    results_dir      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_losses, train_qa, train_qf, train_ret = [], [], [], []
        n_processed = 0

        for batch in train_loader.iter_batches(shuffle=True):
            if len(batch['write_ids']) < batch_size:
                continue
            n_processed += 1
            rng, step_rng = jax.random.split(rng)
            params, opt_state, total_loss, (qa_loss, qf_loss, ret_loss, oracle_loss, acc) = \
                train_step(params, opt_state, batch, step_rng)

            train_losses.append(float(total_loss))
            train_qa.append(float(qa_loss))
            train_qf.append(float(qf_loss))
            train_ret.append(float(ret_loss))

            if np.isnan(float(total_loss)):
                print(f"NaN at epoch {epoch}, batch {n_processed}!")
                raise ValueError("NaN detected in training loss")

        val_losses, val_accs = [], []
        for batch in val_loader.iter_batches(shuffle=False):
            if len(batch['write_ids']) < batch_size:
                continue
            v_loss, v_acc = eval_step(params, batch)
            val_losses.append(float(v_loss))
            val_accs.append(float(v_acc))

        avg_t_loss = np.mean(train_losses)
        avg_v_loss = np.mean(val_losses)   if val_losses else float('inf')
        avg_v_acc  = np.mean(val_accs)     if val_accs   else 0.0

        mark = " ✓ best" if avg_v_loss < best_val_loss else ""
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            best_params   = params
            # Save best checkpoint
            try:
                from flax import serialization
                ckpt_path = os.path.join(results_dir, 'best_model.msgpack')
                with open(ckpt_path, 'wb') as f:
                    f.write(serialization.to_bytes(params))
            except Exception as e:
                print(f"  Checkpoint save failed: {e}")

        print(
            f"Epoch {epoch:03d} | "
            f"Total={avg_t_loss:.4f} QA={np.mean(train_qa):.4f} "
            f"QF={np.mean(train_qf):.4f} Ret={np.mean(train_ret):.4f} | "
            f"Val Loss={avg_v_loss:.4f} Val EM={avg_v_acc*100:.1f}%"
            f"{mark}"
        )

    # ---------------------------------------------------------------------------
    # Test evaluation – FULL test set (no break) – FIX BUG-P0-022
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("TESTING ON FULL TEST SET (GREEDY DECODING)")
    print("=" * 70)

    test_loader = TextDataLoader(test_csv, tokenizer_path, batch_size,
                                 max_input_len, max_target_len, max_samples=None)

    test_losses, test_ems, n_test = [], [], 0

    for batch in test_loader.iter_batches(shuffle=False):
        if len(batch['write_ids']) < batch_size:
            continue
        t_loss, t_em = eval_step(best_params, batch)
        test_losses.append(float(t_loss))
        test_ems.append(float(t_em))
        n_test += batch_size

    avg_test_loss = np.mean(test_losses) if test_losses else float('nan')
    avg_test_em   = np.mean(test_ems)    if test_ems    else float('nan')

    print(f"Test samples evaluated : {n_test}")
    print(f"Test Loss              : {avg_test_loss:.4f}")
    print(f"Test Exact Match       : {avg_test_em * 100:.2f}%")

    # Sample predictions
    print("\n--- Sample Predictions ---")
    for batch in test_loader.iter_batches(shuffle=True):
        if len(batch['write_ids']) < batch_size:
            continue
        mem     = make_blank_memory(config)
        is_eos  = jnp.ones((batch_size,))
        write_p = jnp.ones((batch_size,))

        _, updated_mem = model.apply(
            {'params': best_params, 'memory': mem},
            batch['write_ids'], batch['write_mask'],
            is_eos, write_p, True,
            method=model.write_only,
            mutable=['memory'],
        )

        read_p    = jnp.ones((batch_size,))
        write_off = jnp.zeros((batch_size,))

        preds, _ = model.apply(
            {'params': best_params, 'memory': updated_mem['memory']},
            batch['query_ids'], batch['query_mask'],
            read_p, write_off, max_target_len, 2, pad_id, 3,
            deterministic=True,
            method=model.greedy_decode,
            mutable=['memory'],
        )

        for i in range(min(5, batch_size)):
            q_str = test_loader.tokenizer.decode(batch['query_ids'][i].tolist(), skip_special_tokens=True)
            t_str = test_loader.tokenizer.decode(batch['target_ids'][i].tolist(), skip_special_tokens=True)
            p_str = test_loader.tokenizer.decode(preds[i].tolist(), skip_special_tokens=True)
            print(f"\n[{i+1}] Query:  {q_str}")
            print(f"    Target: {t_str}")
            print(f"    Pred:   {p_str}")
        break  # Only print one batch of examples


if __name__ == '__main__':
    main()
