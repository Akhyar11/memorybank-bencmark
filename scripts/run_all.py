import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import run_benchmark
from scripts import run_ablation

def main():
    print("===========================================")
    print("      STARTING FULL BENCHMARK SUITE        ")
    print("===========================================")
    
    print("\n\n--- BENCHMARKS ---")
    run_benchmark.main()
    
    print("\n\n--- ABLATION ---")
    run_ablation.main()
    
    print("\n===========================================")
    print("      ALL TESTS COMPLETED SUCCESSFULLY     ")
    print("===========================================")

if __name__ == "__main__":
    main()
