import torch
from torch import nn
from typing import Optional, Callable

from ...core import Parameter, Algorithm, jit_class, trace_impl
from ...operators.crossover import simulated_binary
from ...operators.mutation import polynomial_mutation
from ...operators.selection import non_dominate
from ...utils import clamp, nanmin, nanmax, TracingCond
from ...metrics import igd


@jit_class
class NSGA2(Algorithm):

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        pf: torch.Tensor = None,
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

        self.pf = pf

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

        # self.survivor_selection = non_dominate

        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        self.pop = nn.Buffer(population)
        self.fit = nn.Buffer(
            torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf)
        )

    def init_step(self):
        """
        Perform the first optimization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop)


    def step(self):
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
        # print(igd(self.fit, self.pf))
