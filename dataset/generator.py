"""
dataset/generator.py (PyTorch Version)
Generates synthetic data representations for specialized memory experiments.
"""
import torch
import numpy as np


def generate_clustered_dataset(seed, num_samples, dim, num_clusters=50):
    """
    Generates a structured dataset with cluster assignments.
    Returns: (h_eos, cluster_assignments, cluster_centers)
    """
    if isinstance(seed, int):
        torch.manual_seed(seed)
    num_clusters = max(1, min(num_clusters, num_samples // 4))

    # Cluster centers (well-separated on unit sphere)
    cluster_centers = torch.randn(num_clusters, dim)
    cluster_centers = cluster_centers / (torch.norm(cluster_centers, dim=-1, keepdim=True) + 1e-8)

    # Assign samples evenly to clusters
    repeat_count = num_samples // num_clusters + 1
    assignments = torch.from_numpy(
        np.repeat(np.arange(num_clusters), repeat_count)[:num_samples]
    ).long()

    # Add small noise around cluster centers
    noise = torch.randn(num_samples, dim) * 0.15
    h_eos = cluster_centers[assignments] + noise
    h_eos = h_eos / (torch.norm(h_eos, dim=-1, keepdim=True) + 1e-8)
    return h_eos, assignments, cluster_centers


def generate_random_dataset(seed, num_samples, dim):
    """Generates a truly random dataset (uniform on sphere) which is approximately orthogonal in high dims."""
    if isinstance(seed, int):
        torch.manual_seed(seed)
    h_eos = torch.randn(num_samples, dim)
    h_eos = h_eos / (torch.norm(h_eos, dim=-1, keepdim=True) + 1e-8)
    return h_eos


def generate_clustered_pairs(seed, num_samples, dim, num_clusters=50):
    """
    Returns (write_data, query_data) where write and query are from the SAME cluster,
    but query is transformed via a random projection matrix.
    """
    if isinstance(seed, int):
        torch.manual_seed(seed)
    h_write, assignments, cluster_centers = generate_clustered_dataset(seed, num_samples, dim, num_clusters)

    W_raw = torch.randn(dim, dim)
    W, _ = torch.linalg.qr(W_raw)

    projected_centers = torch.matmul(cluster_centers, W)
    query_noise = torch.randn(num_samples, dim) * 0.15
    h_query = projected_centers[assignments] + query_noise
    h_query = h_query / (torch.norm(h_query, dim=-1, keepdim=True) + 1e-8)
    return h_write, h_query


def generate_orthogonal_dataset(seed, num_samples, dim):
    """
    Generates approximately orthogonal h_eos representations.
    Using QR decomposition if num_samples <= dim, otherwise fallback to random normal.
    """
    if isinstance(seed, int):
        torch.manual_seed(seed)
    if num_samples <= dim:
        random_mat = torch.randn(dim, num_samples)
        q, _ = torch.linalg.qr(random_mat)
        return q.T
    else:
        return generate_random_dataset(seed, num_samples, dim)


def add_noise(seed, h_eos, noise_level):
    """Adds Gaussian noise to representations."""
    if isinstance(seed, int):
        torch.manual_seed(seed)
    noise = torch.randn_like(h_eos)
    h_noisy = h_eos + noise_level * noise
    return h_noisy / (torch.norm(h_noisy, dim=-1, keepdim=True) + 1e-8)


def generate_interference_dataset(seed, target_h_eos, num_distractors, noise_level=0.1):
    """Generates distractors by adding noise to a target representation."""
    if isinstance(seed, int):
        torch.manual_seed(seed)
    dim = target_h_eos.shape[-1]
    noise = torch.randn(num_distractors, dim, device=target_h_eos.device)
    distractors = target_h_eos + noise_level * noise
    distractors = distractors / (torch.norm(distractors, dim=-1, keepdim=True) + 1e-8)
    return distractors


def create_synthetic_batch(h_eos):
    """
    Formats the representation into batched inputs suitable for Memory Bank.
    Shape: (batch, dim)
    Returns: h_eos, is_eos, write_prob, read_prob
    """
    batch_size = h_eos.shape[0]
    device = h_eos.device
    is_eos = torch.ones(batch_size, device=device)
    write_prob = torch.ones(batch_size, device=device)
    read_prob = torch.ones(batch_size, device=device)
    return h_eos, is_eos, write_prob, read_prob
