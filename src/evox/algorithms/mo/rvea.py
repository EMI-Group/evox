from typing import Callable, Optional

import torch

from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import ref_vec_guided
from evox.utils import clamp, nanmax, nanmin


class RVEA(Algorithm):
    """
    An implementation of the Reference Vector Guided Evolutionary Algorithm (RVEA) for multi-objective optimization problems.

    This class is designed to solve multi-objective optimization problems using a reference vector guided evolutionary algorithm.

    :references:
        - "A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization," IEEE.
          `Link <https://ieeexplore.ieee.org/document/7386636>`
        - "GPU-accelerated Evolutionary Multiobjective Optimization Using Tensorized RVEA" ACM.
          `Link <https://dl.acm.org/doi/abs/10.1145/3638529.3654223>`
    """

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        alpha: float = 2.0,
        fr: float = 0.1,
        max_gen: int = 100,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        """Initialize the RVEA algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param n_objs: The number of objective functions in the optimization problem.
        :param lb: The lower bounds for the decision variables.
        :param ub: The upper bounds for the decision variables.
        :param alpha: A parameter for controlling the rate of change of penalty. Defaults to 2.
        :param fr: The frequency of reference vector adaptation. Defaults to 0.1.
        :param max_gen: The maximum number of generations. Defaults to 100.
        :param selection_op: The selection operation for evolutionary strategy (optional).
        :param mutation_op: The mutation operation (optional).
        :param crossover_op: The crossover operation (optional).
        :param device: The device on which computations should run (optional).
        """
        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        device = torch.get_default_device() if device is None else device
        # check
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.size(0)
        # write to self
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)

        self.alpha = Parameter(alpha)
        self.fr = Parameter(fr)
        self.max_gen = Parameter(max_gen)

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op
        self.device = device

        if self.selection is None:
            self.selection = ref_vec_guided
        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary
        sampling, _ = uniform_sampling(self.pop_size, self.n_objs)

        v = sampling.to(device=device)

        v0 = v
        self.pop_size = v.size(0)
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.reference_vector = Mutable(v)
        self.init_v = v0
        self.gen = Mutable(torch.tensor(0, dtype=int, device=device))

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop)

    def _rv_adaptation(self, pop_obj: torch.Tensor):
        max_vals = nanmax(pop_obj, dim=0)[0]
        min_vals = nanmin(pop_obj, dim=0)[0]
        return self.init_v * (max_vals - min_vals)

    def _no_rv_adaptation(self, pop_obj: torch.Tensor):
        return self.reference_vector.clone()

    def _mating_pool(self):
        valid_mask = ~torch.isnan(self.pop).all(dim=1)
        num_valid = torch.sum(valid_mask, dtype=torch.int32)
        mating_pool = torch.randint(0, num_valid, (self.pop_size,), device=self.pop.device)
        sorted_indices = torch.where(valid_mask, torch.arange(self.pop_size, device=self.device), torch.inf)
        sorted_indices = torch.argsort(sorted_indices, stable=True)
        pop = self.pop[sorted_indices[mating_pool]]
        return pop

    def _update_pop_and_rv(self, survivor: torch.Tensor, survivor_fit: torch.Tensor):
        self.pop = survivor
        self.fit = survivor_fit

        self.reference_vector = torch.cond(
            self.gen % (1 / self.fr).type(torch.int) == 0, self._rv_adaptation, self._no_rv_adaptation, (survivor_fit,)
        )

    def step(self):
        """Perform a single optimization step."""

        self.gen = self.gen + torch.tensor(1)
        pop = self._mating_pool()
        crossovered = self.crossover(pop)
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        survivor, survivor_fit = self.selection(
            merge_pop,
            merge_fit,
            self.reference_vector,
            (self.gen / self.max_gen) ** self.alpha,
        )

        self._update_pop_and_rv(survivor, survivor_fit)
