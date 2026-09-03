"""
scripts/analyze_accuracy.py – PyTorch Accuracy Analysis
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_predictions import main

if __name__ == '__main__':
    main()
