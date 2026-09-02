import jax
import jax.numpy as jnp
from dataset.generator import generate_random_dataset, create_synthetic_batch
from evaluation.metrics import compute_cosine_similarity
import numpy as np

def run_forgetting(adapter, dim, time_steps, key):
    """
    Test 9: Forgetting / Decay
    Write target. Simulate time steps. Observe if memory becomes dormant/expired.
    """
    key, subkey = jax.random.split(key)
    
    # Generate Target
    target = generate_random_dataset(subkey, 1, dim)
    t_eos, t_is_eos, t_w_prob, _ = create_synthetic_batch(target)
    
    # Write Target
    adapter.reset_memory()
    adapter.write_only(t_eos, t_is_eos, t_w_prob)
    
    # Initial read
    retrieved_v_initial = adapter.read_only(t_eos)
    expected_v = adapter.get_v_proj(t_eos)
    sim_initial = compute_cosine_similarity(retrieved_v_initial, expected_v)
    
    # Simulate time passing by advancing global step directly
    adapter.advance_time(time_steps)
    
    # Trigger decay
    adapter.decay_memory()
    
    # Read again
    retrieved_v_final = adapter.read_only(t_eos)
    sim_final = compute_cosine_similarity(retrieved_v_final, expected_v)
    
    # Retrieve State
    # Need to find the index where it was written. Since it's the first write, it should be index 0
    state = adapter.get_memory_state()[0][0]
    
    return np.mean(np.array(sim_initial)), np.mean(np.array(sim_final)), int(state)

def run_experiment(adapter, config, seeds=3):
    print("Running Forgetting Test...")
    dims = config.memory_dim
    # We want to test t=0, t=10, t=1000, t=10000
    times = [0, 10, 1000, 10000]
    results = {}
    
    for t in times:
        scores_initial = []
        scores_final = []
        states = []
        for seed in range(seeds):
            key = jax.random.PRNGKey(seed)
            si, sf, st = run_forgetting(adapter, dims, t, key)
            scores_initial.append(si)
            scores_final.append(sf)
            states.append(st)
            
        results[t] = {
            'initial': np.mean(scores_initial),
            'final': np.mean(scores_final),
            'state_mode': max(set(states), key=states.count)
        }
        print(f"  T={t}: Initial Sim={results[t]['initial']:.4f}, Final Sim={results[t]['final']:.4f}, State={results[t]['state_mode']}")
        
    return results
