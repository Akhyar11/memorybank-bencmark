"""
models/seed.py
==============
Universal Seeder Mechanism for Deterministic & Reproducible AI Workflows.
Synchronizes Python random, NumPy, PyTorch (CPU + CUDA), and cuDNN backend.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Sets deterministic random seeds across all libraries.
    
    Args:
        seed: Random seed integer (default: 42).
        deterministic: If True, sets cuDNN to deterministic mode.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
