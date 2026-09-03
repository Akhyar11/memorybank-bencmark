"""
scripts/train_text_qa.py – PyTorch End-to-end QA Training script
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.end_to_end_benchmark import run_benchmark

if __name__ == '__main__':
    run_benchmark(num_epochs=10, seeds=[42])
