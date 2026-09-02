import os
import sys
import jax
import jax.numpy as jnp
import optax

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapters.existing_memorybank import MemoryBankAdapter
from dataset.generator import generate_clustered_pairs, create_synthetic_batch


def cosine_loss(pred, target):
    """Cosine similarity loss — robust to scale, prevents trivial zero-output solutions."""
    pred_norm   = pred   / (jnp.linalg.norm(pred,   axis=-1, keepdims=True) + 1e-8)
    target_norm = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    # loss = 1 - cosine_sim  (range: 0 = perfect, 2 = worst)
    return jnp.mean(1.0 - jnp.sum(pred_norm * target_norm, axis=-1))


def train_step(state, batch_write, batch_query, batch_target):
    """
    Proper Memory Bank training:
      1. Write batch_write into memory.
      2. Query memory with batch_query (different samples, same cluster → similar semantics).
      3. Loss: output should be close to batch_target (clean version of batch_write).
    Model CANNOT cheat because query != write. It MUST use retrieval.
    """
    variables, opt_state = state

    def loss_fn(params):
        vars_apply = {'params': params, 'memory': variables['memory']}
        write_inputs, write_eos, write_prob, _ = batch_write
        query_inputs, _, read_prob, _          = batch_query

        # Step 1: Write batch_write into memory
        _, new_vars = adapter.model.apply(
            vars_apply, write_inputs, write_eos, write_prob,
            method=adapter.model.write_only, mutable=['memory']
        )

        # Step 2: Query with batch_query — memory must retrieve batch_write context
        vars_after = {'params': params, 'memory': new_vars['memory']}
        out, vars2 = adapter.model.apply(
            vars_after, query_inputs, read_prob, write_prob, mutable=['memory']
        )

        # Step 3: Loss — output should match batch_target (batch_write)
        target = batch_target[0]  # clean write inputs
        reconstruction_loss = cosine_loss(out, target)

        # Light L2 reg — prevents weights from exploding
        l2 = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        total = reconstruction_loss + 1e-4 * l2
        return total, (vars2['memory'], reconstruction_loss)

    (total, (new_memory, rec_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables['params'])
    updates, new_opt_state = optimizer.update(grads, opt_state, variables['params'])
    new_params = optax.apply_updates(variables['params'], updates)
    new_variables = {'params': new_params, 'memory': new_memory}
    return (new_variables, new_opt_state), rec_loss


def make_paired_batches(data, key, batch_size):
    """
    Split data into (write_batch, query_batch) pairs.
    query_batch has a tiny perturbation to prevent identity shortcut.
    """
    n = (len(data) // (batch_size * 2)) * batch_size * 2
    data = data[:n]
    key, k1, k2 = jax.random.split(key, 3)
    half = n // 2
    write_data = data[:half]
    query_data = data[half:]

    # Small perturbation on query (0.1 noise) — enough to break identity, small enough to be solvable
    noise = jax.random.normal(k2, query_data.shape) * 0.1
    query_data = query_data + noise
    query_data = query_data / (jnp.linalg.norm(query_data, axis=-1, keepdims=True) + 1e-8)

    pairs = []
    for i in range(0, half, batch_size):
        if i + batch_size > half:
            break
        w = write_data[i:i+batch_size]
        q = query_data[i:i+batch_size]
        pairs.append((create_synthetic_batch(w), create_synthetic_batch(q), create_synthetic_batch(w)))
    return pairs, key


if __name__ == "__main__":
    print("Initializing Training...")
    adapter = MemoryBankAdapter()
    adapter.setup()

    learning_rate = 3e-4
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(adapter.variables['params'])

    key = jax.random.PRNGKey(42)
    dataset_size = 6000
    batch_size = 64
    dim = adapter.config.hidden_size

    # Generate same-cluster (write, query) pairs — semantically related
    key, k_train, k_val = jax.random.split(key, 3)
    train_write, train_query = generate_clustered_pairs(k_train, int(dataset_size * 0.8), dim)
    val_write,   val_query   = generate_clustered_pairs(k_val,   int(dataset_size * 0.2), dim)

    def iter_batches(w, q, bs):
        n = (len(w) // bs) * bs
        for i in range(0, n, bs):
            yield create_synthetic_batch(w[i:i+bs]), create_synthetic_batch(q[i:i+bs]), create_synthetic_batch(w[i:i+bs])

    state = (adapter.variables, opt_state)
    train_step_jit = jax.jit(train_step)

    best_val_loss    = float('inf')
    patience         = 8
    patience_counter = 0
    best_state       = state[0]

    print(f"Train: {len(train_write)} | Val: {len(val_write)} | Batch: {batch_size}")
    print(f"Task: Write cluster A → Query perturbed A → Reconstruct A")
    print("-" * 60)

    for epoch in range(100):
        # Shuffle training pairs together
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, len(train_write))
        tw, tq = train_write[perm], train_query[perm]

        train_loss, n_batches = 0.0, 0
        for bw, bq, bt in iter_batches(tw, tq, batch_size):
            state, loss = train_step_jit(state, bw, bq, bt)
            train_loss += float(loss)
            n_batches  += 1
        avg_train = train_loss / max(1, n_batches)

        val_loss, n_val = 0.0, 0
        for bw, bq, bt in iter_batches(val_write, val_query, batch_size):
            _, v = train_step_jit(state, bw, bq, bt)
            val_loss += float(v)
            n_val    += 1
        avg_val = val_loss / max(1, n_val)

        print(f"Epoch {epoch+1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}", end="")

        if avg_val < best_val_loss - 1e-4:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = state[0]
            print("  ✓ best")
        else:
            patience_counter += 1
            print(f"  (no improv {patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}.")
            break

    adapter.variables = best_state
    os.makedirs("results/weights", exist_ok=True)
    adapter.save_weights("results/weights/small_trained.msgpack")
    print(f"\nBest Val Loss: {best_val_loss:.4f}")
    print("Training complete. Weights saved.")

