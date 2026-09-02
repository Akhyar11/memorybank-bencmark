import os
import sys

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.existing_memorybank import MemoryBankAdapter
from experiments import basic_retrieval, persistence, interference, forgetting, replacement, counterfactual
import yaml

def main():
    print("===========================================")
    print("      MEMORY BANK BENCHMARK SUITE          ")
    print("===========================================")
    
    config_path = "configs/small.yaml"
    if not os.path.exists(config_path):
        print(f"Config {config_path} not found.")
        sys.exit(1)
        
    adapter = MemoryBankAdapter(config_path=config_path)
    adapter.setup()
    
    weights_path = "results/weights/small_trained.msgpack"
    if os.path.exists(weights_path):
        adapter.load_weights(weights_path)
    else:
        print(f"WARNING: Weights {weights_path} not found. Running with untrained random weights!")
    
    print("\n[TEST 1] Basic Retrieval")
    basic_retrieval.run_experiment(adapter, adapter.config)
    
    print("\n[TEST 2] Persistence")
    persistence.run_experiment(adapter, adapter.config)
    
    print("\n[TEST 5] Interference")
    interference.run_experiment(adapter, adapter.config)
    
    print("\n[TEST 7] Replacement")
    replacement.run_experiment(adapter, adapter.config)
    
    print("\n[TEST 9] Forgetting")
    forgetting.run_experiment(adapter, adapter.config)
    
    print("\n[TEST 10] Counterfactual")
    counterfactual.run_experiment(adapter, adapter.config)

if __name__ == "__main__":
    main()
