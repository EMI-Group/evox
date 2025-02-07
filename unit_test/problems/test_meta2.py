import os
import time
import unittest
from typing import Callable, Optional

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn.functional as F

from evox.algorithms import PSO
from evox.core import Algorithm, Mutable, Parameter, Problem, jit_class, trace_impl
from evox.metrics import igd
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import non_dominate_rank, ref_vec_guided
from evox.problems.hpo_wrapper import HPOFitnessMonitor, HPOProblemWrapper
from evox.problems.numerical import DTLZ2
from evox.utils import TracingCond, clamp, nanmax, nanmin
from evox.workflows import EvalMonitor, StdWorkflow


@jit_class
class BasicProblem(Problem):
    def __init__(self):
        super().__init__()

    def evaluate(self, x: torch.Tensor):
        return (x * x).sum(-1)


@jit_class
class InnerRVEAa(Algorithm):
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

        self.alpha = alpha
        self.fr = fr
        self.max_gen = max_gen

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
        population0 = torch.rand(self.pop_size, self.dim, device=device)
        population0 = length * population0 + lb
        population = torch.cat([population0, torch.full((self.pop_size, self.dim), torch.nan, device=self.device)], dim=0)
        v = torch.cat([v, torch.rand((self.pop_size, self.n_objs), device=self.device)], dim=0)

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size*2, self.n_objs), device=device).fill_(torch.inf))
        self.reference_vector = Mutable(v)
        self.init_v = v0
        self.ref_vec_init = Parameter(v, device=device)
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
        mating_pool = torch.randint(0, max_idx, (self.pop_size*2,), device=self.device)
        pop_index = torch.where(no_nan_pop, torch.arange(self.pop_size*2, device=self.device), int(1e10))
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
        v_regen = self._rv_regeneration(survivor_fit, self.reference_vector[self.pop_size :])
        self.reference_vector = torch.cat([v_adapt, v_regen], dim=0)

        if self.gen + 1 == self.max_gen:
            self._batch_truncation(survivor, survivor_fit)

        nan_mask_survivor = torch.isnan(survivor).any(dim=1)
        self.pop = survivor[~nan_mask_survivor]
        self.fit = survivor_fit[~nan_mask_survivor]

    @trace_impl(_update_pop_and_rv)
    def _trace_update_pop_and_rv(self, survivor: torch.Tensor, survivor_fit: torch.Tensor):
        state1, names1 = self.prepare_control_flow(self._rv_adaptation, self._no_rv_adaptation)
        if_else1 = TracingCond(self._rv_adaptation, self._no_rv_adaptation)
        state1, v_adapt = if_else1.cond(state1, self.gen % int(1 / self.fr) == 0, survivor_fit)
        self.after_control_flow(state1, *names1)

        v_regen = self._rv_regeneration(survivor_fit, self.reference_vector[self.pop_size: ])
        self.reference_vector = torch.cat([v_adapt, v_regen], dim=0)

        state2, names2 = self.prepare_control_flow(self._batch_truncation, self._no_batch_truncation)
        if_else2 = TracingCond(self._batch_truncation, self._no_batch_truncation)
        state2 = if_else2.cond(state2, self.gen == self.max_gen, survivor, survivor_fit)
        self.after_control_flow(state2, *names2)

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



class solution_transform(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        y = x.view(x.size(0), -1, 3)
        y = y / torch.linalg.vector_norm(y, dim=-1, keepdim=True)
        return {
            "self.algorithm.ref_vec_init": y
        }


class metric(torch.nn.Module):
    def __init__(self, pf: torch.Tensor):
        super().__init__()
        self.pf = pf

    def forward(self, x: torch.Tensor):
        return igd(x, self.pf)


class InnerCore(unittest.TestCase):
    def setUp(
        self, pop_size: int, n_objs: int, dimensions: int, inner_iterations: int, num_instances: int, num_repeats: int = 1
    ):
        self.inner_algo = InnerRVEAa(pop_size=pop_size, n_objs=n_objs, lb=torch.zeros(dimensions), ub=torch.ones(dimensions))
        self.inner_prob = DTLZ2(m=n_objs)
        self.inner_monitor = HPOFitnessMonitor(multi_obj_metric=metric(self.inner_prob.pf()))
        self.inner_workflow = StdWorkflow()
        self.inner_workflow.setup(self.inner_algo, self.inner_prob, monitor=self.inner_monitor)
        self.hpo_prob = HPOProblemWrapper(
            iterations=inner_iterations,
            num_instances=num_instances,
            num_repeats=num_repeats,
            workflow=self.inner_workflow,
            copy_init_state=True,
        )

class OuterCore(unittest.TestCase):
    def setUp(self, num_instances: int, v: torch.Tensor, hpo_prob: HPOProblemWrapper):
        self.outer_algo = PSO(pop_size=num_instances, lb=torch.zeros(v.numel()), ub=torch.ones(v.numel()))
        self.outer_monitor = EvalMonitor(full_sol_history=False)
        self.outer_workflow = StdWorkflow()
        self.outer_workflow.setup(self.outer_algo, hpo_prob, monitor=self.outer_monitor, solution_transform=solution_transform())
        self.outer_workflow.init_step()


if __name__ == "__main__":
    torch.set_default_device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parameters of the inner algorithm
    pop_size = 100
    n_objs = 3
    dimensions = 12

    # Parameters of the hpo problem
    inner_iterations = 100
    num_instances = 1
    num_repeats = 2

    # Iterations of the outer algorithm
    outer_iterations = 100

    # Initialize the inner core
    inner_core = InnerCore()
    inner_core.setUp(
        pop_size=pop_size,
        n_objs=n_objs,
        dimensions=dimensions,
        inner_iterations=inner_iterations,
        num_instances=num_instances,
        num_repeats=num_repeats,
    )
    sampling, _ = uniform_sampling(pop_size, n_objs)
    v = sampling.to()

    # Initialize the outer core
    outer_core = OuterCore()
    outer_core.setUp(v=v, num_instances=num_instances, hpo_prob=inner_core.hpo_prob)

    # params = inner_core.hpo_prob.get_init_params()
    # print("init params:\n", params)

    start_time = time.time()
    for i in range(outer_iterations):
        outer_core.outer_workflow.step()
        if i % 10 == 0:
            print(f"The {i}th iteration and time elapsed: {time.time() - start_time: .4f}(s).")

    outer_monitor = outer_core.outer_workflow.get_submodule("monitor")
    print("params:\n", outer_monitor.topk_solutions, "\n")
    print("result:\n", outer_monitor.topk_fitness)
    # print(outer_monitor.best_fitness)

    # params = inner_core.hpo_prob.get_init_params()
