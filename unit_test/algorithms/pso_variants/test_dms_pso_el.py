import os
import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

import torch

from src.algorithms import DMSPSOEL
from test_base import test


if __name__ == "__main__":
    algo = DMSPSOEL(-10 * torch.ones(1000), 10 * torch.ones(1000), 5000, 9, 5000, max_iteration=1000)
    algo.setup()
    test(algo)