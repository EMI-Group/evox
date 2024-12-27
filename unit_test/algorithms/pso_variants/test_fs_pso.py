import os
import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

import torch

from src.algorithms import FSPSO
from test_base import test


if __name__ == "__main__":
    algo = FSPSO(10000, -10 * torch.ones(1000), 10 * torch.ones(1000))
    algo.setup()
    test(algo)