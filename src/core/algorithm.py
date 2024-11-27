from abc import ABC
from typing import Final

from module import *


class Algorithm(ModuleBase, ABC):
    """Base class for all algorithms"""
    pop_size: Final[int]
    
    def __init__(self, pop_size: int):
        super().__init__()
        assert pop_size > 0, f"Population size shall be larger than 0"
        self.pop_size = pop_size
    
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
    
    def init_tell(self, fitness: torch.Tensor) -> None:
        """Tell the algorithm more information, defaults to `self.tell(fitness)`.

        Workflows only call `init_tell()` for one time, the following tells will all invoke `tell()`.

        Args:
            fitness (`torch.Tensor`): The fitness.
        """
        return self.tell(fitness)
    
    def tell(self, fitness: torch.Tensor) -> None:
        """Tell the algorithm more information.

        Tell the algorithm about the points it chose and their corresponding fitness.

        Args:
            fitness (`torch.Tensor`): The fitness.
        """
        raise NotImplementedError()