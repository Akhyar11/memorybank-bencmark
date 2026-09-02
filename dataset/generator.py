import jax
import jax.numpy as jnp
import numpy as np


def generate_clustered_dataset(key, num_samples, dim, num_clusters=50):
    """
    Generates a structured dataset with cluster assignments.
    Returns: (h_eos, cluster_assignments)
    """
    num_clusters = max(1, min(num_clusters, num_samples // 4))
    key, k1, k2, k3 = jax.random.split(key, 4)

    # Cluster centers (well-separated on unit sphere)
    cluster_centers = jax.random.normal(k1, (num_clusters, dim))
    cluster_centers = cluster_centers / (jnp.linalg.norm(cluster_centers, axis=-1, keepdims=True) + 1e-8)

    # Assign samples evenly to clusters
    assignments = jnp.array(np.repeat(np.arange(num_clusters), num_samples // num_clusters + 1)[:num_samples])

    # Add small noise around cluster centers
    noise = jax.random.normal(k3, (num_samples, dim)) * 0.15
    h_eos = cluster_centers[assignments] + noise
    h_eos = h_eos / (jnp.linalg.norm(h_eos, axis=-1, keepdims=True) + 1e-8)
    return h_eos, assignments


def generate_random_dataset(key, num_samples, dim):
    """Generates structured h_eos via Gaussian Mixture Clusters."""
    h_eos, _ = generate_clustered_dataset(key, num_samples, dim)
    return h_eos


def generate_clustered_pairs(key, num_samples, dim, num_clusters=50):
    """
    Returns (write_data, query_data) where write and query are from the SAME cluster.
    query has a small perturbation so the model can't use identity shortcut.
    """
    key, k1, k2 = jax.random.split(key, 3)
    h_write, assignments = generate_clustered_dataset(k1, num_samples, dim, num_clusters)

    # Generate query = same cluster center + different noise
    num_clusters_actual = int(np.max(np.array(assignments)) + 1)
    cluster_centers = jax.random.normal(k2, (num_clusters_actual, dim))
    cluster_centers = cluster_centers / (jnp.linalg.norm(cluster_centers, axis=-1, keepdims=True) + 1e-8)

    key, k3 = jax.random.split(key)
    query_noise = jax.random.normal(k3, (num_samples, dim)) * 0.15
    h_query = h_write + query_noise  # same cluster, slightly different point
    h_query = h_query / (jnp.linalg.norm(h_query, axis=-1, keepdims=True) + 1e-8)
    return h_write, h_query

def generate_orthogonal_dataset(key, num_samples, dim):
    """
    Generates approximately orthogonal h_eos representations.
    Using QR decomposition if num_samples <= dim, otherwise fallback to random normal.
    """
    if num_samples <= dim:
        random_mat = jax.random.normal(key, (dim, num_samples))
        q, r = jnp.linalg.qr(random_mat)
        return q.T # (num_samples, dim)
    else:
        return generate_random_dataset(key, num_samples, dim)

def add_noise(key, h_eos, noise_level):
    """
    Adds Gaussian noise to representations.
    """
    noise = jax.random.normal(key, h_eos.shape)
    h_noisy = h_eos + noise_level * noise
    return h_noisy / jnp.linalg.norm(h_noisy, axis=-1, keepdims=True)

def generate_interference_dataset(key, target_h_eos, num_distractors, noise_level=0.1):
    """
    Generates distractors by adding noise to a target representation.
    """
    noise = jax.random.normal(key, (num_distractors, target_h_eos.shape[-1]))
    distractors = target_h_eos + noise_level * noise
    distractors = distractors / jnp.linalg.norm(distractors, axis=-1, keepdims=True)
    return distractors

def create_synthetic_batch(h_eos):
    """
    Formats the representation into batched inputs suitable for Memory Bank.
    Shape: (batch, dim)
    Returns: h_eos, is_eos, write_prob, read_prob
    """
    batch_size = h_eos.shape[0]
    is_eos = jnp.ones((batch_size,))
    write_prob = jnp.ones((batch_size,))
    read_prob = jnp.ones((batch_size,))
    return h_eos, is_eos, write_prob, read_prob
