import jax
import jax.numpy as jnp
from dataset.generator import generate_orthogonal_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity
import numpy as np

def run_basic_retrieval(adapter, num_memories, dim, key):
    """
    Test 1: Basic Memory Retrieval
    Writes N memories, then queries one of them.
    """
    key, subkey = jax.random.split(key)
    # Generate orthogonal keys to ensure distinct memories
    dataset = generate_orthogonal_dataset(subkey, num_memories, dim)
    
    # Write all memories
    h_eos, is_eos, write_prob, _ = create_synthetic_batch(dataset)
    
    # One by one write to simulate sequential processing and avoid batch-collapse
    for i in range(num_memories):
        adapter.write_only(h_eos[i:i+1], is_eos[i:i+1], write_prob[i:i+1])
        
    # Query one of them
    target_idx = num_memories // 2
    query_h_eos = h_eos[target_idx:target_idx+1]
    
    # Read
    retrieved_v = adapter.read_only(query_h_eos)
    
    # Expected V
    expected_v = adapter.get_v_proj(query_h_eos)
    
    sim = compute_cosine_similarity(retrieved_v, expected_v)
    
    return np.mean(np.array(sim))

def run_experiment(adapter, config, seeds=5):
    print("Running Basic Retrieval Test...")
    dims = config.memory_dim
    sizes = [10, 50, 100]
    results = {}
    for size in sizes:
        scores = []
        for seed in range(seeds):
            key = jax.random.PRNGKey(seed)
            # Re-init adapter for fresh state
            adapter.reset_memory()
            score = run_basic_retrieval(adapter, size, dims, key)
            scores.append(score)
        results[size] = {'mean': np.mean(scores), 'std': np.std(scores)}
        print(f"  Size {size}: Sim = {results[size]['mean']:.4f} ± {results[size]['std']:.4f}")
    return results
