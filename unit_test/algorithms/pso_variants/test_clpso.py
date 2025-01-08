import os
import sys
current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

import torch

from src.algorithms import CLPSO
from unit_test.algorithms.test_base import test


if __name__ == "__main__":
    algo = CLPSO(pop_size=10000, lb=-10 * torch.ones(1000), ub=10 * torch.ones(1000))
    algo.setup()
    test(algo)