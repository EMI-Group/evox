from abc import ABC
from typing import Any

import torch

from ..core.module import ModuleBase


class Algorithm(ModuleBase, ABC):
    """Base class for all algorithms

    ## Notice:
    If a subclass have defined `trace_impl` of `step`, its corresponding `init_step` must be overwritten even though nothing special is to be included due to Python's object-oriented limitations.
    """

    def __init__(self):
        super().__init__()

    def step(self) -> None:
        """Execute the algorithm procedure for one step."""
        pass

    def init_step(self) -> None:
        """Initialize the algorithm and execute the algorithm procedure for the first step."""
        self.step()

    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        """Evaluate the fitness at given points.
        This function is a proxy function of `Problem.evaluate` set by workflow.
        By default, this functions raises `NotImplementedError`.

        Args:
            pop (`torch.Tensor` or any): The population.

        Returns:
            `torch.Tensor`: The fitness.
        """
        raise NotImplementedError(
            "Evaluate function is not implemented. It is a proxy function of `Problem.evaluate` set by workflow."
        )


class Problem(ModuleBase, ABC):
    """Base class for all problems"""

    def __init__(self):
        super().__init__()

    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        """Evaluate the fitness at given points

        Args:
            pop (`torch.Tensor` or any): The population.

        Returns:
            `torch.Tensor`: The fitness.

        ## Notice:
        If this function contains external evaluations that cannot be JIT by `torch.jit`, please wrap it with `torch.jit.ignore`.
        """
        return torch.empty(0)


class Workflow(ModuleBase, ABC):
    """The base class for workflow."""

    def init_step(self) -> None:
        """Perform the first optimization step of the workflow."""
        return self.step()

    def step(self) -> None:
        """The basic function to step a workflow."""
        pass


class Monitor(ModuleBase, ABC):
    """
    The monitor base class.

    Monitors are used to monitor the evolutionary process.
    They contains a set of callbacks,
    which will be called at specific points during the execution of the workflow.
    Monitor itself lives outside the main workflow, so jit is not required.

    To implements a monitor, implement your own callbacks and override the hooks method.
    The hooks method should return a list of strings, which are the names of the callbacks.
    Currently the supported callbacks are:

    `post_ask`, `pre_eval`, `post_eval`, and `pre_tell`.
    """

    def set_config(self, **config) -> "Monitor":
        """Set the static variables according to `config`.

        Args:
            config: The configuration.

        Returns:
            This module.
        """
        return self

    def post_ask(self, candidate_solution: torch.Tensor) -> None:
        """The hook function to be executed before the solution transformation.

        Args:
            candidate_solution (`torch.Tensor`): The population (candidate solutions) before the solution transformation.
        """
        pass

    def pre_eval(self, transformed_candidate_solution: Any) -> None:
        """The hook function to be executed after the solution transformation.

        Args:
            transformed_candidate_solution (`torch.Tensor` or any): The population (candidate solutions) after the solution transformation.
        """
        pass

    def post_eval(self, fitness: torch.Tensor) -> None:
        """The hook function to be executed before the fitness transformation.

        Args:
            fitness (`torch.Tensor`): The fitnesses before the fitness transformation.
        """
        pass

    def pre_tell(self, transformed_fitness: torch.Tensor) -> None:
        """The hook function to be executed after the fitness transformation.

        Args:
            transformed_fitness (`torch.Tensor`): The fitnesses after the fitness transformation.
        """
        pass
