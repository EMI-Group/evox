from abc import ABC
import sys
from typing import Optional, Final, Union, Any

sys.path.append(__file__ + "/../..")

import torch
from torch import nn

from core.module import ModuleBase, jit_class, trace_impl, use_state
from core.jit_util import vmap, jit


class Algorithm(ModuleBase, ABC):
    """Base class for all algorithms

    ## Notice:
    If a subclass have defined `trace_impl` of `ask` or `tell`, its corresponding `init_ask` or `init_tell` must be overwritten even though nothing special is to be included, because Python cannot correctly find the `trace_impl` version of these function due to otherwise.
    """

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
        pass

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
        pass


class Problem(ModuleBase, ABC):
    """Base class for all problems"""

    num_obj: Final[int]

    def __init__(self, num_objective: int):
        super().__init__()
        assert num_objective > 0, f"Number of objectives shall be larger than 0"
        self.num_obj = num_objective

    def evaluate(self, pop: torch.Tensor | Any) -> torch.Tensor:
        """Evaluate the fitness at given points

        Args:
            pop (`torch.Tensor` or any): The population.

        Returns:
            `torch.Tensor`: The fitness.

        ## Notice:
        If this function contains external evaluations that cannot be JIT by `torch.jit`, please wrap it with `torch.jit.ignore`.
        """
        pass

    def eval(self):
        assert False, "Problem.eval() shall never be invoked to prevent ambiguity."


class Workflow(ModuleBase, ABC):
    """The base class for workflow."""

    def step(self) -> None:
        """The basic function to step a workflow.

        Usually consists of sequence invocation of `algorithm.ask()`, `problem.evaluate()`, and `algorithm.tell()`.
        """
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

    `pre_step`, `post_step`, `pre_ask`, `post_ask`, `pre_eval`, `post_eval`, `pre_tell`, `post_tell`, and `post_step`.
    """

    def set_config(self, **config):
        """Set the static variables according to `config`.

        Args:
            config: The configuration.
            
        Returns:
            This module.
        """
        return self

    def pre_step(self):
        pass

    def pre_ask(self):
        pass

    def post_ask(self, candidate_solution: Union[torch.Tensor, Any]):
        pass

    def pre_eval(self, transformed_candidate_solution: Union[torch.Tensor, Any]):
        pass

    def post_eval(self, fitness: torch.Tensor):
        pass

    def pre_tell(self, transformed_fitness: torch.Tensor):
        pass

    def post_tell(self):
        pass

    def post_step(self):
        pass