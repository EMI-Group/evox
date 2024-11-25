from abc import ABC
from typing import Union, Any

from .module import *


class Problem(ModuleBase, ABC):
    """Base class for all problems"""
    
    def evaluate(self, pop: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Evaluate the fitness at given points

        Args:
            pop (`torch.Tensor` or any): The population.

        Returns:
            `torch.Tensor`: The fitness.
        """
        raise NotImplementedError()
