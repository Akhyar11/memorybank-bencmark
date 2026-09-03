"""
scripts/demo_interference.py – PyTorch Demo Interference
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.interference import run_experiment

if __name__ == '__main__':
    run_experiment()
