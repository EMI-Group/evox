from typing import Callable, Optional

import torch
import torch.nn.functional as F

from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import non_dominate_rank, ref_vec_guided
from evox.utils import clamp, nanmax, nanmin, randint


class RVEAa(Algorithm):
    """
    An implementation of the Reference Vector Guided Evolutionary Algorithm embedded with the reference vector
    regeneration strategy (RVEAa) for multi-objective optimization problems.

    This class is designed to solve multi-objective optimization problems using a reference vector guided evolutionary algorithm.

    :references:
        [1] R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, "A reference vector guided evolutionary algorithm
            for many-objective optimization," IEEE Transactions on Evolutionary Computation, vol. 20, no. 5,
            pp. 773-791, 2016. Available: https://ieeexplore.ieee.org/document/7386636
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
        """Initialize the RVEAa algorithm with the given parameters.

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
        self.lb = lb.unsqueeze(0).to(device=device)
        self.ub = ub.unsqueeze(0).to(device=device)

        self.alpha = Parameter(alpha)
        self.fr = Parameter(fr)
        self.max_gen = Parameter(max_gen)

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

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

        length = self.ub - self.lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + self.lb

        v1 = torch.rand(self.pop_size, self.n_objs, device=device)
        v = torch.cat([v, v1], dim=0)

        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.reference_vector = Mutable(v.clone())
        self.init_v = v0.clone()

        self.gen = Mutable(torch.tensor(0, dtype=torch.long, device=device))
        self.rv_adapt_every = Mutable(torch.tensor(1, dtype=torch.long, device=device))

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        rv_adapt_every = torch.round(1.0 / self.fr).to(device=self.pop.device)
        rv_adapt_every = torch.clamp(rv_adapt_every, min=1)
        self.rv_adapt_every = rv_adapt_every.to(dtype=torch.long)

        self.fit = self.evaluate(self.pop)

    def _rv_adaptation(self, pop_obj: torch.Tensor):
        max_vals = nanmax(pop_obj, dim=0)[0]
        min_vals = nanmin(pop_obj, dim=0)[0]
        return self.init_v * (max_vals - min_vals)

    def _no_rv_adaptation(self, pop_obj: torch.Tensor):
        return self.reference_vector[: self.pop_size].clone()

    def _mating_pool(self):
        valid_mask = ~torch.isnan(self.pop).all(dim=1)
        num_valid = torch.sum(valid_mask, dtype=torch.int32)

        mating_pool = randint(0, num_valid, (self.pop_size,), device=self.pop.device)

        sorted_indices = torch.where(
            valid_mask,
            torch.arange(self.pop.size(0), device=self.pop.device),
            torch.iinfo(torch.int32).max,
        )
        sorted_indices = torch.argsort(sorted_indices, stable=True)
        pop = self.pop[sorted_indices[mating_pool]]
        return pop

    def _rv_regeneration(
        self, pop_obj: torch.Tensor, v: torch.Tensor, rank: torch.Tensor
    ):
        """Reference-vector regeneration, compile-safe rewrite (no dynamic shapes).

        Semantics identical to the original: aux reference vectors with no
        associated non-dominated solution are replaced by random vectors scaled
        to the ND-front extent; the rest are kept. All fixed-shape ops
        (``torch.where`` / one-hot scatter), boolean/advanced indexing removed —
        mirrors the verified vmap-safe rewrite in MetaRVEA's InnerRVEAa.
        ``rank`` is passed in (computed once in ``_update_pop_and_rv``) instead
        of being recomputed on a variable-length subset here.
        """
        valid_mask = ~torch.isnan(pop_obj).all(dim=1)
        nd_mask = (rank == 0) & valid_mask
        nd_obj = torch.where(nd_mask.unsqueeze(1), pop_obj, torch.nan)

        min_vals = nanmin(nd_obj, dim=0).values
        max_vals = nanmax(nd_obj, dim=0).values
        nd_shifted = nd_obj - min_vals

        cosine = F.cosine_similarity(
            nd_shifted.unsqueeze(1), v.unsqueeze(0), dim=-1
        )
        cosine = torch.where(torch.isnan(cosine), -torch.inf, cosine)
        associate = cosine.max(dim=1).indices

        assoc_onehot = (
            associate.unsqueeze(1)
            == torch.arange(v.size(0), device=pop_obj.device).unsqueeze(0)
        ).float()
        assoc_onehot = assoc_onehot * nd_mask.float().unsqueeze(1)
        counts = assoc_onehot.sum(dim=0)

        scale = (max_vals - min_vals).clamp(min=1e-6).unsqueeze(0)
        rand = torch.rand_like(v) * scale
        return torch.where((counts == 0).unsqueeze(1), rand, v)

    def _batch_truncation(
        self,
        pop: torch.Tensor,
        obj: torch.Tensor,
        rank: torch.Tensor,
    ):
        """Final-generation ND truncation, compile-safe rewrite.

        Semantics identical to the original: keep only rank-0 (non-dominated)
        rows and NaN-pad the rest. Implemented with fixed-shape ops instead of
        boolean indexing, mirroring InnerRVEAa's verified rewrite. ``rank`` is
        passed in (computed once in ``_update_pop_and_rv``).
        """
        valid_mask = ~torch.isnan(obj).all(dim=1)
        nd_mask = (rank == 0) & valid_mask
        new_pop = torch.where(nd_mask.unsqueeze(1), pop, torch.nan)
        new_obj = torch.where(nd_mask.unsqueeze(1), obj, torch.nan)
        return new_pop, new_obj

    def _no_batch_truncation(self, pop: torch.Tensor, obj: torch.Tensor):
        return pop.clone(), obj.clone()

    def _update_pop_and_rv(self, survivor: torch.Tensor, survivor_fit: torch.Tensor):
        v_aux = self.reference_vector[self.pop_size :]
        # rank 计算一次复用：原实现 _rv_regeneration/_batch_truncation 各自对
        # 同一输入重算（valid 行 NaN→inf 逐元素映射，两者逐位等价）
        obj_safe = torch.where(torch.isnan(survivor_fit), torch.inf, survivor_fit)
        rank = non_dominate_rank(obj_safe)
        v_regen = self._rv_regeneration(survivor_fit, v_aux, rank)

        # torch.where 替代 Python if（数据依赖分支在 fullgraph 下 graph-break）
        do_adapt = (self.gen % self.rv_adapt_every) == 0
        v_adapt_on = self._rv_adaptation(survivor_fit)
        v_adapt_off = self._no_rv_adaptation(survivor_fit)
        v_adapt = torch.where(do_adapt, v_adapt_on, v_adapt_off)

        # 两分支都算 + where 选择；末代 ND 截断与非末代直通
        trunc_pop, trunc_obj = self._batch_truncation(survivor, survivor_fit, rank)
        no_pop, no_obj = self._no_batch_truncation(survivor, survivor_fit)
        is_final = self.gen == self.max_gen
        self.pop = torch.where(is_final, trunc_pop, no_pop)
        self.fit = torch.where(is_final, trunc_obj, no_obj)

        self.reference_vector = torch.cat([v_adapt, v_regen], dim=0)

    def step(self):
        """Perform a single optimization step."""

        self.gen = self.gen + torch.tensor(1, dtype=self.gen.dtype, device=self.gen.device)

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