"""
evaluation/metrics.py – Fixed evaluation metrics.

Fixes applied:
- BUG-P1-009: recall_at_k now implements proper Recall@K with ranked list
- Added MRR (Mean Reciprocal Rank)
- Added token F1
- Added exact match
"""
import numpy as np


def compute_cosine_similarity(retrieved, target):
    """
    Cosine similarity between two batched vectors.
    Args:
        retrieved: (batch, dim) or (dim,)
        target:    (batch, dim) or (dim,)
    Returns:
        sim: (batch,) or scalar
    """
    retrieved = np.array(retrieved)
    target    = np.array(target)
    if retrieved.ndim == 1:
        retrieved = retrieved[None, :]
        target    = target[None, :]
        squeeze   = True
    else:
        squeeze   = False

    r_norm = retrieved / (np.linalg.norm(retrieved, axis=-1, keepdims=True) + 1e-8)
    t_norm = target    / (np.linalg.norm(target,    axis=-1, keepdims=True) + 1e-8)
    sim    = np.sum(r_norm * t_norm, axis=-1)
    return sim[0] if squeeze else sim


def recall_at_k(scores, ground_truth_idx=None, k_values=(1, 3, 5), gt_idx=None):
    """
    Recall@K for a single query against ranked scores.

    Args:
        scores:           (capacity,) – retrieval scores for all slots
        ground_truth_idx: int – index of the ground truth slot
        k_values:         list of K values to evaluate

    Returns:
        dict: {1: 0/1, 3: 0/1, 5: 0/1} – whether GT is in top-K
    """
    if ground_truth_idx is None:
        ground_truth_idx = gt_idx
    scores = np.array(scores)
    if len(scores) == 0 or ground_truth_idx is None:
        return {k: 0.0 for k in k_values}

    sorted_indices = np.argsort(scores)[::-1]  # descending

    results = {}
    for k in k_values:
        top_k = sorted_indices[:k]
        results[k] = 1.0 if ground_truth_idx in top_k else 0.0
    return results


def mean_reciprocal_rank(scores, ground_truth_idx=None, gt_idx=None):
    """
    Compute reciprocal rank of the ground truth slot.

    Args:
        scores:           (capacity,) – retrieval scores
        ground_truth_idx: int

    Returns:
        float: 1/rank if found, else 0.0
    """
    if ground_truth_idx is None:
        ground_truth_idx = gt_idx
    scores = np.array(scores)
    if len(scores) == 0 or ground_truth_idx is None:
        return 0.0

    sorted_indices = np.argsort(scores)[::-1]  # descending
    ranks = np.where(sorted_indices == ground_truth_idx)[0]
    if len(ranks) == 0:
        return 0.0
    return float(1.0 / (ranks[0] + 1))


def exact_match(predictions, targets, pad_id=0):
    """
    Exact match score: all non-pad tokens must match exactly.

    Args:
        predictions: (batch, seq_len) int array
        targets:     (batch, seq_len) int array
        pad_id:      padding token id

    Returns:
        float: fraction of sequences with exact match
    """
    predictions = np.array(predictions)
    targets     = np.array(targets)
    mask        = (targets != pad_id)
    tok_match   = (predictions == targets) | (~mask)
    seq_match   = np.all(tok_match, axis=-1)
    return float(np.mean(seq_match))


def token_f1(prediction, target, pad_id=0):
    """
    Token-level F1 for a single sequence (like SQuAD F1).

    Args:
        prediction: list of token ids
        target:     list of token ids
        pad_id:     padding token id

    Returns:
        float: F1 score
    """
    pred_tokens = [t for t in prediction if t != pad_id]
    true_tokens = [t for t in target     if t != pad_id]

    if len(pred_tokens) == 0 and len(true_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return 0.0

    pred_set = set(pred_tokens)
    true_set = set(true_tokens)
    common   = pred_set & true_set

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(true_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return f1


def batch_token_f1(predictions, targets, pad_id=0):
    """
    Average token F1 over a batch.

    Args:
        predictions: (batch, seq_len) int array
        targets:     (batch, seq_len) int array
        pad_id:      padding id

    Returns:
        float: mean F1
    """
    predictions = np.array(predictions)
    targets     = np.array(targets)
    scores      = [
        token_f1(predictions[i].tolist(), targets[i].tolist(), pad_id)
        for i in range(len(predictions))
    ]
    return float(np.mean(scores))


def cosine_sim_threshold_recall(retrieved_vals, target_val, threshold=0.9):
    """
    Fraction of retrievals with cosine similarity above threshold.
    (Backward-compatible with old recall_at_k usage.)
    """
    sim   = compute_cosine_similarity(retrieved_vals, target_val)
    match = np.array(sim) > threshold
    return float(np.mean(match))
