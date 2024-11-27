from abc import ABC
from typing import Final, Union, Any

from module import *


class Problem(ModuleBase, ABC):
    """Base class for all problems"""
    num_obj: Final[int]
    
    def __init__(self, num_objective: int):
        super().__init__()
        assert num_objective > 0, f"Number of objectives shall be larger than 0"
        self.num_obj = num_objective
    
    def evaluate(self, pop: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Evaluate the fitness at given points

        Args:
            pop (`torch.Tensor` or any): The population.

        Returns:
            `torch.Tensor`: The fitness.
        """
        raise NotImplementedError()
    
    def eval(self):
        assert False, "Problem.eval() shall never be invoked to prevent ambiguity."