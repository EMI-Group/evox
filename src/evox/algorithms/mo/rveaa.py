from typing import Callable, Optional

import torch
import torch.nn.functional as F

from ...core import Algorithm, Mutable, Parameter, jit_class, trace_impl, debug_print
from ...operators.crossover import simulated_binary
from ...operators.mutation import polynomial_mutation
from ...operators.sampling import uniform_sampling
from ...operators.selection import non_dominate_rank, ref_vec_guided
from ...utils import TracingCond, clamp, nanmax, nanmin


@jit_class
class RVEAa(Algorithm):
    """RVEAa的PyTorch实现"""

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
        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        if device is None:
            device = torch.get_default_device()
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
        v0 = v.clone()
        self.pop_size = v.size(0)
        length = ub - lb
        population0 = torch.rand(self.pop_size, self.dim, device=device)
        population0 = length * population0 + lb
        population = torch.cat([population0, torch.full((self.pop_size, self.dim), torch.nan, device=self.device)], dim=0)
        v = torch.cat([v, torch.rand((self.pop_size, self.n_objs), device=self.device)], dim=0)

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size*2, self.n_objs), device=device).fill_(torch.inf))
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
        return self.reference_vector[: self.pop_size]

    def _mating_pool(self):
        mating_pool = torch.randint(0, self.pop.size(0), (self.pop_size,))
        return self.pop[mating_pool]

    @trace_impl(_mating_pool)
    def _trace_mating_pool(self):
        no_nan_pop = ~torch.isnan(self.pop).all(dim=1)
        max_idx = torch.sum(no_nan_pop, dtype=torch.int32)
        mating_pool = torch.randint(0, max_idx, (self.pop_size * 2,), device=self.device)
        pop_index = torch.where(no_nan_pop, torch.arange(self.pop_size * 2, device=self.device), int(1e10))
        pop_index = torch.argsort(pop_index, stable=True)
        pop = self.pop[pop_index[mating_pool].squeeze()]
        return pop

    def _rv_regeneration(self, pop_obj: torch.Tensor, v: torch.Tensor):
        """Regenerate reference vectors strategy (PyTorch版本)"""
        pop_obj = pop_obj - nanmin(pop_obj, dim=0).values
        cosine = F.cosine_similarity(pop_obj.unsqueeze(1), v.unsqueeze(0), dim=-1)
        associate = nanmax(cosine, dim=1).indices
        invalid = torch.sum((associate.unsqueeze(1) == torch.arange(v.size(0), device=pop_obj.device)), dim=0)
        rand = torch.rand((v.size(0), v.size(1)), device=pop_obj.device) * nanmax(pop_obj, dim=0).values
        v = torch.where((invalid == 0).unsqueeze(1), rand, v)

        return v

    def _batch_truncation(self, pop: torch.Tensor, obj: torch.Tensor):
        n = pop.size(0) // 2
        cosine = F.cosine_similarity(obj.unsqueeze(1), obj.unsqueeze(0), dim=-1)
        not_all_nan_rows = ~torch.isnan(cosine).all(dim=1)
        mask = torch.eye(cosine.size(0), dtype=torch.bool, device=pop.device) & not_all_nan_rows.unsqueeze(1)
        cosine = torch.where(mask, torch.as_tensor(0.0, device=pop.device), cosine)

        sorted_values, _ = torch.sort(-cosine, dim=1)
        sorted_values = torch.where(torch.isnan(sorted_values[:, 0]), -torch.inf, sorted_values[:, 0])
        rank = torch.argsort(sorted_values)

        mask = torch.ones(rank.size(0), dtype=torch.bool, device=pop.device)
        # mask[rank[:n]] = 0
        mask = torch.where(torch.arange(rank.size(0), device=pop.device) < n, torch.tensor(0, dtype=torch.bool, device=pop.device), mask)
        mask = mask.unsqueeze(1)

        new_pop = torch.where(mask, pop, torch.nan)
        new_obj = torch.where(mask, obj, torch.nan)

        self.pop = new_pop
        self.fit = new_obj

    def _no_batch_truncation(self, pop: torch.Tensor, obj: torch.Tensor):
        self.pop = pop
        self.fit = obj

    def _update_pop_and_rv(self, survivor: torch.Tensor, survivor_fit: torch.Tensor):
        if self.gen % (1 / self.fr).type(torch.int) == 0:
            v_adapt = self._rv_adaptation(survivor_fit)
        else:
            v_adapt = self._no_rv_adaptation(survivor_fit)
        # v_regen = self._rv_regeneration(survivor_fit, self.reference_vector[self.pop_size :])
        v_regen = self.reference_vector[self.pop_size :]
        self.reference_vector = torch.cat([v_adapt, v_regen], dim=0)

        if self.gen == self.max_gen:
            self._batch_truncation(survivor, survivor_fit)

    @trace_impl(_update_pop_and_rv)
    def _trace_update_pop_and_rv(self, survivor: torch.Tensor, survivor_fit: torch.Tensor):
        state1, names1 = self.prepare_control_flow(self._rv_adaptation, self._no_rv_adaptation)
        if_else1 = TracingCond(self._rv_adaptation, self._no_rv_adaptation)
        state1, v_adapt = if_else1.cond(state1, self.gen % (1 / self.fr).type(torch.int) == 0, survivor_fit)
        self.after_control_flow(state1, *names1)

        # v_regen = self._rv_regeneration(survivor_fit, self.reference_vector[self.pop_size :])
        v_regen = self.reference_vector[self.pop_size :]
        self.reference_vector = torch.cat([v_adapt, v_regen], dim=0)

        state2, names2 = self.prepare_control_flow(self._batch_truncation, self._no_batch_truncation)
        if_else2 = TracingCond(self._batch_truncation, self._no_batch_truncation)
        state2 = if_else2.cond(state2, self.gen == self.max_gen, survivor, survivor_fit)
        self.after_control_flow(state2, *names2)

    def step(self):
        """Perform a single optimization step."""

        self.gen = self.gen + torch.tensor(1, device=self.device)
        pop = self._mating_pool()
        crossovered = self.crossover(pop)
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        rank = non_dominate_rank(merge_fit)
        merge_fit = torch.where(rank.unsqueeze(1) == 0, merge_fit, torch.nan)
        merge_pop = torch.where(rank.unsqueeze(1) == 0, merge_pop, torch.nan)

        survivor, survivor_fit = self.selection(
            merge_pop,
            merge_fit,
            self.reference_vector,
            (self.gen / self.max_gen) ** self.alpha,
        )

        self._update_pop_and_rv(survivor, survivor_fit)
