import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

import torch

from src.algorithms import XNES, SeparableNES
from unit_test.algorithms.test_base import test


if __name__ == "__main__":
    algo = XNES(pop_size=10000, init_mean=torch.rand(1000) * 20 - 10, init_covar = torch.eye(1000) )
    algo.setup()
    test(algo)
    
    algo = SeparableNES(pop_size=10000, init_mean=torch.rand(1000) * 20 - 10, init_std = torch.full( (1000,),1) )
    algo.setup()
    test(algo)
