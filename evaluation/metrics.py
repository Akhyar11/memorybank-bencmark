import jax.numpy as jnp
import numpy as np

def compute_cosine_similarity(retrieved, target):
    """
    Computes cosine similarity between two batched vectors.
    """
    retrieved_norm = retrieved / (jnp.linalg.norm(retrieved, axis=-1, keepdims=True) + 1e-8)
    target_norm = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    sim = jnp.sum(retrieved_norm * target_norm, axis=-1)
    return sim

def recall_at_k(retrieved_vals, target_val, k_vals=None):
    """
    Given that we are retrieving fused representations or values directly,
    measuring exact match is hard in continuous space.
    We measure success if the cosine similarity of retrieved value to target value > threshold.
    Alternatively, if the model returns a set of candidates, we could do traditional recall.
    Since fusion returns a single vector `fused = W_f[h; m]`, we might want to evaluate `m` directly.
    We will modify the test adapter to return `m` for clean evaluation.
    """
    # For now, we will evaluate if the retrieved vector is highly correlated with the target
    sim = compute_cosine_similarity(retrieved_vals, target_val)
    # Binary match if sim > 0.9
    match = sim > 0.9
    return np.mean(np.array(match))
