# BUG_AUDIT_V2.md – Deep Audit V2

Generated from: Second deep audit (2026-09-02)

## P0 – Fatal Architecture / Benchmark Failures

### BUG-V2-P0-001
- **File**: `models/tiny_memory_bank.py`
- **Function**: `read()`
- **Problem**: Read gating only zeros the output but still applies reinforcement (access count, last access, importance boost) to the retrieved slots.
- **Why it is wrong**: If `read_prob = 0` (read disabled), the read operation must have no side-effects on memory state.
- **Scientific impact**: Memories get falsely reinforced even when the model decides not to read, corrupting the memory state.
- **Technical impact**: `access_count`, `last_access`, `importance` are incorrectly incremented.
- **Fix**: Apply the `is_read` mask (or `read_prob` > 0 condition) when computing access reinforcement.
- **How to test**: Add `test_read_gate()` that asserts state remains identical when `read_prob=0`.

### BUG-V2-P0-002
- **File**: `models/tiny_memory_bank.py`
- **Function**: `__init__` or `setup()`
- **Problem**: No explicit `empty_memory_state()` method.
- **Why it is wrong**: The initialization relies on `jnp.zeros` which happens to align with `STATE_EXPIRED = 0`. An explicit method is needed to guarantee correct semantics (e.g. `confidence=0`, etc.).
- **Scientific impact**: State initialization is implicit and could be broken by changes in configuration.
- **Technical impact**: Difficult to reset memory correctly in episodic evaluation.
- **Fix**: Create `empty_memory_state(batch_size)` or explicit reset logic.
- **How to test**: Assert that initial state matches exact defined empty semantics (0 keys/vals, 0 importance, EXPIRED state, etc.).

### BUG-V2-P0-003
- **File**: `models/tiny_memory_bank.py`
- **Function**: `read()`
- **Problem**: Top-K retrieval aggregates inactive slots if `active_slots < top_k`.
- **Why it is wrong**: Inactive (EXPIRED) slots contribute to the softmax denominator, diluting retrieval scores of valid slots.
- **Scientific impact**: Retrieval is mathematically incorrect when memory is mostly empty.
- **Technical impact**: Softmax over `-1e9` for invalid slots can cause numerical issues or incorrect weighting.
- **Fix**: Ensure mask completely excludes EXPIRED slots from softmax sum, and top-k correctly ignores them.
- **How to test**: Test 2 active memories with `top_k = 8` and verify inactive slots contribute exactly zero.

### BUG-V2-P0-004
- **File**: `dataset/generator.py`
- **Function**: `generate_clustered_pairs()`
- **Problem**: Query is generated as `h_write + query_noise` which makes the query nearly identical to the stored key (identity shortcut).
- **Why it is wrong**: A memory benchmark must not allow trivial copy shortcuts.
- **Scientific impact**: Overestimates memory capability because the task doesn't require general association.
- **Technical impact**: Distractor retrieval tests are too easy.
- **Fix**: Derive query from the same cluster structure but ensure it is distinct from `h_write`.
- **How to test**: Assert cosine similarity between query and key is bounded and not near 1.0.

## P1 – Major Correctness Issues

### BUG-V2-P1-001
- **File**: `tests/test_memory_functional.py`
- **Function**: `test_recall_at_1_with_distractors()`
- **Problem**: Test always passes via `assert ... or True`.
- **Why it is wrong**: False positive tests hide failures.
- **Scientific impact**: Distractor resilience is claimed but not proven.
- **Technical impact**: Broken code can pass CI.
- **Fix**: Assert actual retrieval similarity or Recall@1 metric.
- **How to test**: Run test with bad params and ensure it fails.

### BUG-V2-P1-002
- **File**: `tests/test_memory_functional.py`
- **Function**: `TestInterference.test_target_retrievable_after_interference()`
- **Problem**: Only checks `jnp.linalg.norm(out) > 1e-6`.
- **Why it is wrong**: This only proves *something* was retrieved, not that the *correct* target survived interference.
- **Scientific impact**: Interference robustness is unverified.
- **Technical impact**: Test is meaningless.
- **Fix**: Measure exact match or degradation against baseline retrieval.
- **How to test**: Ensure test fails if distractor count is massively increased or decay is aggressive.

### BUG-V2-P1-003
- **File**: `tests/test_counterfactual.py`
- **Function**: `test_different_values_produce_different_retrievals()`
- **Problem**: Intervention negates `v_proj` weights. This changes how ALL values are processed, not a true isolated causal intervention on one slot.
- **Why it is wrong**: The counterfactual test must isolate ONE intervention (changing the stored value in the slot) while keeping retrieval mechanism identical.
- **Scientific impact**: Does not cleanly prove causality of the stored value.
- **Technical impact**: Weak test.
- **Fix**: Intervene directly on the stored memory value (`vals`), keeping all projections and keys identical.
- **How to test**: Assert that output changes exactly corresponding to the changed value.

### BUG-V2-P1-004
- **File**: `dataset/text_dataset_loader.py`
- **Function**: `iter_batches()`
- **Problem**: Variable length batches at the end of the dataset might be dropped or padded incorrectly without `valid_count`.
- **Why it is wrong**: The full test set must be evaluated.
- **Scientific impact**: Results are not strictly over the full evaluation set.
- **Technical impact**: Metric inflation/deflation.
- **Fix**: Track valid elements in partial batches and mask them in metrics.
- **How to test**: Assert evaluated sample count matches exact dataset size.

## P2 – Moderate Issues

### BUG-V2-P2-001
- **File**: `models/tiny_memory_bank.py`
- **Function**: `decay_memory()`
- **Problem**: Test does not rigorously verify exact state transitions (e.g. boundary conditions).
- **Why it is wrong**: Only verifies one point.
- **Fix**: Add test for multiple dt thresholds.
- **How to test**: Assert transition EXPIRED vs DORMANT precisely at `effective_R = 0.1` and `0.5`.

### BUG-V2-P2-002
- **File**: `tests/test_memory_state.py`
- **Problem**: Recency, importance, confidence tests only check that the value increases, not that it correctly influences retrieval score.
- **Fix**: Add causal tests where only one of these factors varies, and prove it changes the retrieval ranking.
- **How to test**: Set two memories with identical properties except importance, verify the high importance one is retrieved.
