from typing import Callable, Optional

import torch

from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import non_dominate_rank, tournament_selection
from evox.utils import clamp, lexsort


def cal_hv(fit: torch.Tensor, ref: torch.Tensor, pop_size: int, n_sample: int):
    n, m = fit.size()
    alpha = torch.cumprod(torch.cat([torch.ones(1, device=fit.device), (pop_size - torch.arange(1, n, device=fit.device)) / (n - torch.arange(1, n, device=fit.device))]), dim=0) / torch.arange(1, n + 1, device=fit.device)
    alpha = torch.nan_to_num(alpha)

    f_min = torch.min(fit, dim=0).values

    samples = torch.rand(n_sample, m, device=fit.device) * (ref - f_min) + f_min

    ds = torch.zeros(n_sample, dtype=torch.int64, device=fit.device)
    pds = (fit.unsqueeze(0).expand(n_sample, -1, -1) - samples.unsqueeze(1).expand(-1, n, -1) <= 0).all(dim=2)
    ds = torch.sum(torch.where(pds, ds.unsqueeze(1) + 1, ds.unsqueeze(1)), dim=1)
    ds = torch.where(ds == 0, ds, ds - 1)

    temp = torch.where(pds.T, ds.unsqueeze(0), -1)
    value = torch.where(temp != -1, alpha[temp], torch.tensor(0, dtype=torch.float32))
    f = torch.sum(value, dim=1)

    f = f * torch.prod(ref - f_min) / n_sample
    return f


class HypE(Algorithm):
    """The tensoried version of HypE algorithm.

    :reference: https://direct.mit.edu/evco/article-abstract/19/1/45/1363/HypE-An-Algorithm-for-Fast-Hypervolume-Based-Many
    """

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        n_sample: int = 10000,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        """Initializes the HypE algorithm.

        :param pop_size: The size of the population.
        :param n_objs: The number of objective functions in the optimization problem.
        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param n_sample: The number of samples for hypervolume calculation (optional).
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
        self.n_sample = n_sample

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        self.selection = tournament_selection
        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary

        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        self.ref = Mutable(torch.ones(n_objs, device=device))

        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))


    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop)
        self.ref = torch.full((self.n_objs,), torch.max(self.fit).item() * 1.2, device=self.fit.device)

    def step(self):
        """Perform the optimization step of the workflow."""
        hv = cal_hv(self.fit, self.ref, self.pop_size, self.n_sample)
        mating_pool = self.selection(self.pop_size, -hv)
        crossovered = self.crossover(self.pop[mating_pool])
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)

        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        rank = non_dominate_rank(merge_fit)
        order = torch.argsort(rank)
        worst_rank = rank[order[self.pop_size - 1]]
        mask = rank <= worst_rank

        hv = cal_hv(merge_fit, self.ref, torch.sum(mask) - self.pop_size, self.n_sample)
        dis = torch.where(mask, hv, -torch.inf)

        combined_indices = lexsort([-dis, rank])[: self.pop_size]

        self.pop = merge_pop[combined_indices]
        self.fit = merge_fit[combined_indices]
