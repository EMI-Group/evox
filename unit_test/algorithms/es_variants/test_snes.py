import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

import torch

from src.algorithms import SNES
from unit_test.algorithms.test_base import test


if __name__ == "__main__":
    algo = SNES(pop_size=10000, center_init=torch.rand(1000) * 20 - 10, weight_type= "recomb" )
    algo.setup()
    test(algo)
    
    algo = SNES(pop_size=10000, center_init=torch.rand(1000) * 20 - 10, weight_type= "temp" )
    algo.setup()
    test(algo)

