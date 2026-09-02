import jax
import jax.numpy as jnp
from dataset.generator import generate_orthogonal_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity
import numpy as np

def run_replacement(adapter, capacity, dim, over_capacity, key):
    """
    Test 7: Replacement
    Write C memories. Write over_capacity additional memories. Check if the first ones were replaced.
    """
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Generate C initial memories
    initial_dataset = generate_orthogonal_dataset(subkey1, capacity, dim)
    h_init, is_init, w_init, _ = create_synthetic_batch(initial_dataset)
    
    # Generate additional memories
    new_dataset = generate_orthogonal_dataset(subkey2, over_capacity, dim)
    h_new, is_new, w_new, _ = create_synthetic_batch(new_dataset)
    
    adapter.reset_memory()
    
    # Write initial C memories
    for i in range(capacity):
        adapter.write_only(h_init[i:i+1], is_init[i:i+1], w_init[i:i+1])
        
    # Read initial memory 0 before replacement
    retrieved_before = adapter.read_only(h_init[0:1])
    sim_before = compute_cosine_similarity(retrieved_before, adapter.get_v_proj(h_init[0:1]))
    
    # Write additional memories
    for i in range(over_capacity):
        adapter.write_only(h_new[i:i+1], is_new[i:i+1], w_new[i:i+1])
        
    # Read initial memory 0 after replacement
    retrieved_after = adapter.read_only(h_init[0:1])
    sim_after = compute_cosine_similarity(retrieved_after, adapter.get_v_proj(h_init[0:1]))
    
    # Count how many of the initial memories are still retrievable
    retrievable_count = 0
    for i in range(capacity):
        r = adapter.read_only(h_init[i:i+1])
        s = compute_cosine_similarity(r, adapter.get_v_proj(h_init[i:i+1]))
        if np.mean(s) > 0.8:
            retrievable_count += 1
            
    return np.mean(np.array(sim_before)), np.mean(np.array(sim_after)), retrievable_count

def run_experiment(adapter, config, seeds=3):
    print("Running Replacement Test...")
    capacity = config.memory_capacity
    dims = config.memory_dim
    
    # We will test over_capacity = 10, capacity/2, capacity
    over_sizes = [10, capacity // 2, capacity]
    results = {}
    
    for size in over_sizes:
        scores_before = []
        scores_after = []
        counts = []
        for seed in range(seeds):
            key = jax.random.PRNGKey(seed)
            sb, sa, count = run_replacement(adapter, capacity, dims, size, key)
            scores_before.append(sb)
            scores_after.append(sa)
            counts.append(count)
            
        results[size] = {
            'before': np.mean(scores_before),
            'after': np.mean(scores_after),
            'survival_rate': np.mean(counts) / capacity
        }
        print(f"  Over Capacity {size}: Sim Before={results[size]['before']:.4f}, Sim After={results[size]['after']:.4f}, Survival Rate={results[size]['survival_rate']*100:.1f}%")
        
    return results
