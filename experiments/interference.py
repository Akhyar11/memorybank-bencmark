import jax
import jax.numpy as jnp
from dataset.generator import generate_random_dataset, generate_interference_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity
import numpy as np

def run_interference(adapter, num_distractors, dim, noise_level, key):
    """
    Test 5: Interference
    Write target, write distractors that are highly similar to target. Read target.
    """
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Generate Target
    target = generate_random_dataset(subkey1, 1, dim)
    t_eos, t_is_eos, t_w_prob, _ = create_synthetic_batch(target)
    
    # Write Target
    adapter.write_only(t_eos, t_is_eos, t_w_prob)
    
    # Generate and Write Distractors (similar to target)
    if num_distractors > 0:
        distractors = generate_interference_dataset(subkey2, target, num_distractors, noise_level)
        d_eos, d_is_eos, d_w_prob, _ = create_synthetic_batch(distractors)
        for i in range(num_distractors):
            # Interference: These might trigger 'update' instead of 'insert' depending on threshold!
            adapter.write_only(d_eos[i:i+1], d_is_eos[i:i+1], d_w_prob[i:i+1])
            
    # Read Target
    retrieved_v = adapter.read_only(t_eos)
    expected_v = adapter.get_v_proj(t_eos)
    
    sim = compute_cosine_similarity(retrieved_v, expected_v)
    return np.mean(np.array(sim))

def run_experiment(adapter, config, seeds=3):
    print("Running Interference Test...")
    dims = config.memory_dim
    sizes = [10, 50]
    noise_levels = [0.1, 0.5]
    results = {}
    for noise in noise_levels:
        for size in sizes:
            scores = []
            for seed in range(seeds):
                key = jax.random.PRNGKey(seed)
                adapter.reset_memory()
                score = run_interference(adapter, size, dims, noise, key)
                scores.append(score)
            results[(noise, size)] = {'mean': np.mean(scores), 'std': np.std(scores)}
            print(f"  Noise {noise}, Distractors {size}: Sim = {results[(noise, size)]['mean']:.4f} ± {results[(noise, size)]['std']:.4f}")
    return results
