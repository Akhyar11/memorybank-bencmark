import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.existing_memorybank import MemoryBankAdapter
from experiments import ablation

def main():
    print("===========================================")
    print("      MEMORY BANK ABLATION SUITE           ")
    print("===========================================")
    
    config_path = "configs/small.yaml"
    if not os.path.exists(config_path):
        print(f"Config {config_path} not found.")
        sys.exit(1)
        
    ablation.run_experiment(config_path, None)

if __name__ == "__main__":
    main()
