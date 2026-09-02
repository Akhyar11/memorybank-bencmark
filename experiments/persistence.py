import jax
import jax.numpy as jnp
from dataset.generator import generate_random_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity
import numpy as np

def run_persistence(adapter, num_distractors, dim, key):
    """
    Test 2: Memory Persistence
    Write target, write N distractors, read target.
    """
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Generate Target
    target = generate_random_dataset(subkey1, 1, dim)
    t_eos, t_is_eos, t_w_prob, _ = create_synthetic_batch(target)
    
    # Write Target
    adapter.write_only(t_eos, t_is_eos, t_w_prob)
    
    # Generate and Write Distractors
    if num_distractors > 0:
        distractors = generate_random_dataset(subkey2, num_distractors, dim)
        d_eos, d_is_eos, d_w_prob, _ = create_synthetic_batch(distractors)
        for i in range(num_distractors):
            adapter.write_only(d_eos[i:i+1], d_is_eos[i:i+1], d_w_prob[i:i+1])
            
    # Read Target
    retrieved_v = adapter.read_only(t_eos)
    expected_v = adapter.get_v_proj(t_eos)
    
    sim = compute_cosine_similarity(retrieved_v, expected_v)
    return np.mean(np.array(sim))

def run_experiment(adapter, config, seeds=3):
    print("Running Persistence Test...")
    dims = config.memory_dim
    # Don't exceed capacity for basic persistence test unless we are testing replacement
    max_distractors = config.memory_capacity - 2 
    sizes = [0, 10, min(100, max_distractors), max_distractors]
    results = {}
    for size in sizes:
        scores = []
        for seed in range(seeds):
            key = jax.random.PRNGKey(seed)
            adapter.reset_memory()
            score = run_persistence(adapter, size, dims, key)
            scores.append(score)
        results[size] = {'mean': np.mean(scores), 'std': np.std(scores)}
        print(f"  Distractors {size}: Sim = {results[size]['mean']:.4f} ± {results[size]['std']:.4f}")
    return results
