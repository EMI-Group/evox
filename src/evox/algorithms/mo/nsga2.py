import torch
from torch import nn
from typing import Optional, Callable

from ...core import Algorithm, Mutable, jit_class, trace_impl
from ...operators.crossover import simulated_binary
from ...operators.mutation import polynomial_mutation
from ...operators.selection import non_dominate
from ...utils import clamp
from ...metrics import igd


@jit_class
class NSGA2(Algorithm):
    """
    An implementation of the Non-dominated Sorting Genetic Algorithm II (NSGA-II) for multi-objective optimization problems.

    This class provides a framework for solving multi-objective optimization problems using a non-dominated sorting genetic algorithm,
    which is widely used for Pareto-based optimization.

    :references:
        - "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," IEEE Transactions on Evolutionary Computation.
          `Link <https://ieeexplore.ieee.org/document/996017>`_
    """
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        """ Initializes the NSGA-II algorithm.

         :param pop_size: The size of the population.
         :param n_objs: The number of objective functions in the optimization problem.
         :param lb: The lower bounds for the decision variables (1D tensor).
         :param ub: The upper bounds for the decision variables (1D tensor).
         :param selection_op: The selection operation for evolutionary strategy (optional).
         :param mutation_op: The mutation operation, defaults to `polynomial_mutation` if not provided (optional).
         :param crossover_op: The crossover operation, defaults to `simulated_binary` if not provided (optional).
         :param device: The device on which computations should run (optional). Defaults to PyTorch's default device.
         """

        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        if device is None:
            device = torch.get_default_device()
        # check
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.shape[0]
        # write to self
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op
        self.device = device

        # if self.selection is None:
        #     self.selection = ref_vec_guided
        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary

        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        self.pop = Mutable(population)
        self.fit = Mutable(
            torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf)
        )

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop)

    def step(self):
        """Perform the optimization step of the workflow.
        """
        crossovered = self.crossover(self.pop)
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        self.pop, self.fit = non_dominate(
            merge_pop,
            merge_fit,
            self.pop_size
        )
