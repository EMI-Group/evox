import time
import unittest
from typing import Callable, Optional

import torch

from evox.algorithms import PSO
from evox.core import Algorithm, Mutable, Parameter, Problem, jit_class, trace_impl
from evox.metrics import igd
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import ref_vec_guided
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
class InnerRVEA(Algorithm):
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
        """Initialize the MetaRVEA algorithm with the given parameters. This algorithm should be the inner algorithm of a HPO problem, using the reference vector as the hyperparameter.

        :param pop_size: The size of the population.
        :param n_objs: The number of objective functions in the optimization problem.
        :param lb: The lower bounds for the decision variables.
        :param ub: The upper bounds for the decision variables.
        :param alpha: A parameter for controlling the rate of change of penalty. Defaults to 2. In general, alpha is a hyperparameter.
        :param fr: The frequency of reference vector adaptation. Defaults to 0.1. In general, fr is a hyperparameter.
        :param max_gen: The maximum number of generations. Defaults to 100. In general, max_gen is a hyperparameter.]
        :param selection_op: The selection operation for evolutionary strategy (optional).
        :param mutation_op: The mutation operation (optional).
        :param crossover_op: The crossover operation (optional).
        :param device: The device on which computations should run (optional).
        """
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
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))

        self.reference_vector = Mutable(v)
        self.init_v = v0
        self.ref_vec_init = Parameter(v0, device=device)

        self.gen = Mutable(torch.tensor(0, dtype=int, device=device))

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.reference_vector = torch.as_tensor(self.ref_vec_init)
        self.fit = self.evaluate(self.pop)

    def _rv_adaptation(self, pop_obj: torch.Tensor):
        max_vals = nanmax(pop_obj, dim=0)[0]
        min_vals = nanmin(pop_obj, dim=0)[0]
        return self.init_v * (max_vals - min_vals)

    def _no_rv_adaptation(self, pop_obj: torch.Tensor):
        return self.reference_vector

    def _mating_pool(self):
        mating_pool = torch.randint(0, self.pop.size(0), (self.pop_size,))
        return self.pop[mating_pool]

    @trace_impl(_mating_pool)
    def _trace_mating_pool(self):
        no_nan_pop = ~torch.isnan(self.pop).all(dim=1)
        max_idx = torch.sum(no_nan_pop, dtype=torch.int32)
        mating_pool = torch.randint(0, max_idx, (self.pop_size,), device=self.device)
        pop_index = torch.where(no_nan_pop, torch.arange(self.pop_size), torch.inf)
        pop_index = torch.argsort(pop_index, stable=True)
        pop = self.pop[pop_index[mating_pool].squeeze()]
        return pop

    def _update_pop_and_rv(self, survivor: torch.Tensor, survivor_fit: torch.Tensor):
        nan_mask_survivor = torch.isnan(survivor).any(dim=1)
        self.pop = survivor[~nan_mask_survivor]
        self.fit = survivor_fit[~nan_mask_survivor]

        if self.gen % (1 / self.fr).int() == 0:
            self.reference_vector = self._rv_adaptation(survivor_fit)

    @trace_impl(_update_pop_and_rv)
    def _trace_update_pop_and_rv(self, survivor: torch.Tensor, survivor_fit: torch.Tensor):
        state, names = self.prepare_control_flow(self._rv_adaptation, self._no_rv_adaptation)
        if_else = TracingCond(self._rv_adaptation, self._no_rv_adaptation)
        state, reference_vector = if_else.cond(state, self.gen % int(1 / self.fr) == 0, survivor_fit)
        self.after_control_flow(state, *names)
        self.reference_vector = reference_vector
        self.pop = survivor
        self.fit = survivor_fit

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
        self.inner_algo = InnerRVEA(pop_size=pop_size, n_objs=n_objs, lb=-torch.zeros(dimensions), ub=torch.ones(dimensions))
        self.inner_prob = DTLZ2(m=n_objs)
        self.inner_monitor = HPOFitnessMonitor(multi_obj_metric=metric(self.inner_prob.pf()))
        # self.inner_monitor = HPOFitnessMonitor(multi_obj_metric=metric(self.inner_prob.pf()),fit_aggregation=lambda x, dim: torch.min(x, dim=dim)[0])
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
    inner_iterations = 1000
    num_instances = 10
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
