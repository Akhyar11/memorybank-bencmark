"""
scripts/ablation_test.py – Redirect wrapper to PyTorch experiments.ablation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.ablation import run_experiment

if __name__ == '__main__':
    run_experiment()
