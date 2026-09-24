"""Fix random seeds so reruns draw the same random numbers."""
import random

import numpy as np
import torch

SEED = 0


def set_seeds(seed: int = SEED) -> None:
    """Seed Python's `random`, NumPy's global generator, and PyTorch on every device (incl. MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)