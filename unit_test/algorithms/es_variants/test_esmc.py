import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

import torch

from src.algorithms import ESMC
from unit_test.algorithms.test_base import test


if __name__ == "__main__":
    algo = ESMC(pop_size=10001, center_init=torch.rand(1000) * 20 - 10 )
    algo.setup()
    test(algo)
    
    algo = ESMC(pop_size=10001, center_init=torch.rand(1000) * 20 - 10, optimizer="adam" )
    algo.setup()
    test(algo)