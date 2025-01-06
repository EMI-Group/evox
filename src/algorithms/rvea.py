import torch
from torch import nn
from typing import Optional, Callable

from ..core import Parameter, Algorithm, jit_class, trace_impl
from ..operators import (
    simulated_binary,
    uniform_sampling,
    polynomial_mutation,
    ref_vec_guided,
)
from ..utils import clamp, nanmin, nanmax, TracingCond
from ..metrics import igd


@jit_class
class RVEA(Algorithm):
    """
    An implementation of the Reference Vector Guided Evolutionary Algorithm (RVEA) for multi-objective optimization problems.

    This class is designed to solve multi-objective optimization problems using a reference vector guided evolutionary algorithm .

    References:
        - "A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization," IEEE.
          [Link](https://ieeexplore.ieee.org/document/7386636)
        - "GPU-accelerated Evolutionary Multiobjective Optimization Using Tensorized RVEA" ACM.
          [Link](https://dl.acm.org/doi/abs/10.1145/3638529.3654223)

    Attributes:
        pop_size (int): The size of the population.
        n_objs (int): The number of objective functions in the optimization problem.
        lb (torch.Tensor): The lower bounds for the decision variables.
        ub (torch.Tensor): The upper bounds for the decision variables.
        pf (torch.Tensor | None): The Pareto front for reference (optional).
        alpha (float): A parameter for controlling the rate of change of penalty. Defaults to 2.
        fr (float): The frequency of reference vector adaptation. Defaults to 0.1.
        max_gen (int): The maximum number of generations. Defaults to 100.
        selection_op (Callable | None): The selection operation for evolutionary strategy (optional).
        mutation_op (Callable | None): The mutation operation (optional).
        crossover_op (Callable | None): The crossover operation (optional).
        device (torch.device | None): The device on which computations should run (optional).
    """

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        pf: torch.Tensor = None,
        alpha: float = 2,
        fr: float = 0.1,
        max_gen: int = 100,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
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

        self.alpha = Parameter(alpha)
        self.fr = Parameter(fr)
        self.max_gen = Parameter(max_gen)
        self.pf = pf

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
        self.pop_size = v.shape[0]
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        self.pop = nn.Buffer(population)
        self.fit = nn.Buffer(
            torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf)
        )
        self.reference_vector = Parameter(v)  # nn.Buffer(v)
        self.init_v = nn.Buffer(v0)
        self.gen = nn.Buffer(torch.tensor(0, device=device))

    def init_step(self):
        """
        Perform the first optimization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop)

    def rv_adaptation(self, pop_obj: torch.Tensor, v0: torch.Tensor, v: torch.Tensor):
        max_vals = nanmax(pop_obj, dim=0)[0]
        min_vals = nanmin(pop_obj, dim=0)[0]
        return v0 * (max_vals - min_vals)

    def _mating_pool(self):
        mating_pool = torch.randint(0, self.pop.shape[0], (self.pop_size,))
        return self.pop[mating_pool]

    @trace_impl(_mating_pool)
    def _trace_mating_pool(self):
        no_nan_pop = ~torch.isnan(self.pop).all(dim=1)
        max_idx = torch.sum(no_nan_pop, dtype=torch.int32)
        mating_pool = torch.randint(0, max_idx, (self.pop_size,), device=self.device)
        pop = self.pop[torch.nonzero(no_nan_pop)[mating_pool].squeeze()]
        return pop

    def _update_pop_and_rv(self, survivor: torch.Tensor, survivor_fit: torch.Tensor):
        nan_mask_survivor = torch.isnan(survivor).any(dim=1)
        self.pop = survivor[~nan_mask_survivor]
        self.fit = survivor_fit[~nan_mask_survivor]

        if self.gen % (1 / self.fr) == 0:
            self.reference_vector = self.rv_adaptation(
                survivor_fit, self.init_v, self.reference_vector
            )

    @trace_impl(_update_pop_and_rv)
    def _trace_update_pop_and_rv(
        self, survivor: torch.Tensor, survivor_fit: torch.Tensor
    ):
        self.pop = survivor
        self.fit = survivor_fit

        if_eles = TracingCond(
            lambda x, y, z: [self.rv_adaptation(x, y, z)], lambda x, y, z: [z]
        )
        self.reference_vector = if_eles.cond(
            self.gen % (1 / self.fr) == 0, self.fit, self.init_v, self.reference_vector
        )[0]

    def step(self):
        self.gen += 1
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
            torch.tensor(
                (self.gen / self.max_gen) ** self.alpha, device=merge_fit.device
            ),
        )

        self._update_pop_and_rv(survivor, survivor_fit)
