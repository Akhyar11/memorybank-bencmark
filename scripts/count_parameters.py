"""
scripts/count_parameters.py – Parameter count report.

Reports:
- Embedding parameters
- Memory Bank parameters (learned projections only, not memory state)
- Encoder parameters
- Decoder parameters
- Total parameters
- Memory state tensors (runtime state, not trainable)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from models.tiny_memory_bank import TinyMemoryConfig
from models.transformer_qa_model import TransformerQAModel


def count_params(params, prefix=""):
    """Recursively count parameters in a nested dict."""
    total = 0
    for k, v in params.items():
        if isinstance(v, dict):
            total += count_params(v, prefix=f"{prefix}.{k}")
        else:
            n = int(np.prod(v.shape))
            total += n
    return total


def count_params_by_module(params):
    """Count params grouped by top-level module name."""
    counts = {}
    for k, v in params.items():
        counts[k] = count_params(v) if isinstance(v, dict) else int(np.prod(v.shape))
    return counts


def report(config=None, vocab_size=2000, embed_dim=32, num_layers=1, num_heads=2, ff_dim=64, max_target_len=16):
    if config is None:
        config = TinyMemoryConfig(
            memory_capacity=128, memory_dim=32, hidden_size=32,
            memory_top_k=8,
        )

    model = TransformerQAModel(
        config=config, vocab_size=vocab_size, embed_dim=embed_dim,
        num_layers=num_layers, num_heads=num_heads, ff_dim=ff_dim,
        max_target_len=max_target_len, dropout_rate=0.0,
    )

    rng   = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 16), dtype=jnp.int32)
    dp    = jnp.ones((1,), dtype=jnp.float32)
    dt    = jnp.ones((1, max_target_len), dtype=jnp.int32)

    variables = model.init(rng, dummy, dummy, dp, dp, dt, method=model.init_all)
    params    = variables['params']
    memory    = variables['memory']  # runtime state

    by_module = count_params_by_module(params)

    # Group counts
    embed_n   = by_module.get('embedding', 0)
    bank_n    = by_module.get('bank', 0)
    enc_fact  = sum(v for k, v in by_module.items() if 'fact_encoders' in k)
    enc_query = sum(v for k, v in by_module.items() if 'query_encoders' in k)
    dec_n     = sum(v for k, v in by_module.items() if 'decoder' in k)
    pos_n     = by_module.get('pos_encoding', 0)
    other_n   = sum(v for k, v in by_module.items()
                    if k not in ('embedding', 'bank')
                    and 'encoder' not in k and 'decoder' not in k
                    and 'pos_encoding' not in k)

    total_params = count_params(params)

    # Memory state sizes (runtime, not trained)
    mem_state_n = sum(int(np.prod(v.shape)) for v in memory.values()
                      if isinstance(v, jnp.ndarray))

    print("=" * 55)
    print("  PARAMETER COUNT REPORT")
    print("=" * 55)
    print(f"  Embedding parameters:         {embed_n:>10,}")
    print(f"  Memory Bank (learned proj):   {bank_n:>10,}")
    print(f"  Fact Encoder parameters:      {enc_fact:>10,}")
    print(f"  Query Encoder parameters:     {enc_query:>10,}")
    print(f"  Decoder parameters:           {dec_n:>10,}")
    print(f"  Positional Encoding:          {pos_n:>10,}")
    print(f"  Other parameters:             {other_n:>10,}")
    print(f"  {'─'*40}")
    print(f"  TOTAL TRAINABLE PARAMETERS:   {total_params:>10,}")
    print(f"")
    print(f"  Memory State (runtime only):  {mem_state_n:>10,}")
    print(f"    (not trained, resets episodically)")
    print("=" * 55)

    # Detailed bank breakdown
    print("\n  Memory Bank Detail:")
    bank_params = params.get('bank', {})
    for name, arr in bank_params.items():
        if isinstance(arr, dict):
            n = count_params(arr)
            print(f"    {name:20s}: {n:>8,} params")
        else:
            print(f"    {name:20s}: {int(np.prod(arr.shape)):>8,} params")

    return {
        'embedding':    embed_n,
        'memory_bank':  bank_n,
        'encoder':      enc_fact + enc_query,
        'decoder':      dec_n,
        'total':        total_params,
        'memory_state': mem_state_n,
    }


if __name__ == '__main__':
    report()
