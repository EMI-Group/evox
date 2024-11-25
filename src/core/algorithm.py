from abc import ABC
from typing import Any

from .module import *


class Algorithm(ModuleBase, ABC):
    """Base class for all algorithms"""
    
    def init_ask(self) -> torch.Tensor:
        """Ask the algorithm for the first time, defaults to `self.ask()`.

        Workflows only call `init_ask()` for one time, the following asks will all invoke `ask()`.

        Returns:
            `torch.Tensor`: The initial candidate solution.
        """
        return self.ask()
    
    def ask(self) -> torch.Tensor:
        """Ask the algorithm.

        Ask the algorithm for points to explore.

        Returns:
            `torch.Tensor`: The candidate solution.
        """
        raise NotImplementedError()
    
    def init_tell(self, fitness: torch.Tensor) -> torch.Tensor:
        """Tell the algorithm more information, defaults to `self.tell(fitness)`.

        Workflows only call `init_tell()` for one time, the following tells will all invoke `tell()`.

        Args:
            fitness (`torch.Tensor`): The fitness.
        """
        return self.init_tell(fitness)
    
    def tell(self, fitness: torch.Tensor) -> None:
        """Tell the algorithm more information.

        Tell the algorithm about the points it chose and their corresponding fitness.

        Args:
            fitness (`torch.Tensor`): The fitness.
        """
        raise NotImplementedError()
