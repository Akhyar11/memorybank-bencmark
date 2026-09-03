"""
scripts/analyze_sim.py – PyTorch Cosine Similarity Analysis
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.basic_retrieval import run_experiment

if __name__ == '__main__':
    run_experiment()
