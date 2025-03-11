# Outer algorithm: A variant of PSO
# Inner algorithm: RVEAa
# Inner problem: DLTZ7
# Hyperparameters: all instances' random reference vectors
# Strategy: optimize all instances' random reference vectors but only change one reference vector of every instance during the last half iterations
# Result of n_objs = 3 (IGD): 0.0465
# Result of n_objs = 5 (IGD): 0.2102
# Result of n_objs = 10 (IGD): 0.8401
# Result of n_objs = 10 and outer iterations = 1000 (IGD): 0.8364
# Result of n_objs = 15 (IGD): 1.7866

import time
import unittest
from typing import Callable, Optional

import test_meta2
import test_meta3
import test_meta4
import test_meta5
import torch
import torch.nn.functional as F

from evox.algorithms import PSO
from evox.algorithms.pso_variants.utils import min_by
from evox.core import Algorithm, Mutable, Parameter, jit_class, trace_impl
from evox.metrics import igd
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import non_dominate_rank, ref_vec_guided
from evox.problems.hpo_wrapper import HPOFitnessMonitor, HPOProblemWrapper
from evox.problems.numerical import DTLZ7
from evox.utils import TracingCond, clamp, clamp_float, maximum, nanmax, nanmin
from evox.workflows import EvalMonitor, StdWorkflow


@jit_class
class OuterPSO(Algorithm):
    """The basic Particle Swarm Optimization (PSO) algorithm.

    ## Class Methods

    * `__init__`: Initializes the PSO algorithm with given parameters (population size, lower and upper bounds, inertia weight, cognitive weight, and social weight).
    * `step`: Performs a single optimization step using Particle Swarm Optimization (PSO), updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    """

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        inner_pop_size: int,
        n_objs: int,
        max_gen: int = 100,
        w: float = 0.6,
        phi_p: float = 2.5,
        phi_g: float = 0.8,
        device: torch.device | None = None,
    ):
        """
        Initialize the PSO algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param w: The inertia weight. Defaults to 0.6.
        :param phi_p: The cognitive weight. Defaults to 2.5.
        :param phi_g: The social weight. Defaults to 0.8.
        :param lb: The lower bounds of the particle positions. Must be a 1D tensor.
        :param ub: The upper bounds of the particle positions. Must be a 1D tensor.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype
        self.pop_size = pop_size
        self.dim = lb.shape[0]
        # use inner_pop_size and n_objs to adjust PSO and add an additional dimension
        self.inner_pop_size = inner_pop_size
        self.n_objs = n_objs
        self.max_gen = max_gen
        # Here, Parameter is used to indicate that these values are hyper-parameters
        # so that they can be correctly traced and vector-mapped
        self.w = Parameter(w, device=device)
        self.phi_p = Parameter(phi_p, device=device)
        self.phi_g = Parameter(phi_g, device=device)
        # setup
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb
        velocity = torch.rand(self.pop_size, self.inner_pop_size, self.n_objs, device=device)
        velocity = 2 * length * velocity - length
        # write to self
        self.lb = lb
        self.ub = ub
        # mutable
        self.population = Mutable(population)
        self.velocity = Mutable(velocity)

        real_population = self.population.view(self.pop_size, self.inner_pop_size, self.n_objs)

        self.local_best_location = Mutable(real_population)
        self.local_best_fitness = Mutable(torch.empty(self.pop_size,self.inner_pop_size, device=device).fill_(torch.inf))
        self.global_best_location = Mutable(real_population[:,0])
        self.global_best_fitness = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))

        self.gen = Mutable(torch.tensor(0, dtype=int, device=device))

    def _velocity_process(self, velocity: torch.Tensor):
        # Add mask for most individuals, except the individuals with the largest norm of (self.global_best_location - self.population)
        distance_instance = self.global_best_location - self.population
        distance_vector = distance_instance.view(self.pop_size, -1, self.n_objs)
        mask = torch.zeros_like(distance_vector, dtype=torch.bool, device=velocity.device)
        global_norm = torch.norm(distance_vector, p=2, dim=2)
        largest_indices = nanmax(global_norm, dim=1).indices.unsqueeze(-1).unsqueeze(-1)
        mask.scatter_(dim=1, index=largest_indices.expand(-1, -1, 3), value=1)
        # print(mask.size())
        mask = mask.view(self.pop_size, -1)
        # print(mask.size())
        velocity = torch.where(mask, velocity, torch.zeros_like(velocity, device=velocity.device))
        # debug_print("velocity{}: ",velocity[0])
        return velocity

    def _no_velocity_process(self, velocity: torch.Tensor):
        print(self.gen)
        return velocity

    def _mask_velocity(self, velocity: torch.Tensor):
        if self.gen<=self.max_gen/2:
            return self._velocity_process(velocity)
        return velocity

    @trace_impl(_mask_velocity)
    def _trace_mask_velocity(self, velocity: torch.Tensor):
        state, names = self.prepare_control_flow(self._velocity_process, self._no_velocity_process)
        if_else = TracingCond(self._velocity_process, self._no_velocity_process)
        state, velocity = if_else.cond(state, self.gen<=self.max_gen/2, velocity)
        self.after_control_flow(state, *names)
        return velocity

    def step(self):
        """
        Perform a normal optimization step using PSO.

        This function evaluates the fitness of the current population, updates the
        local best positions and fitness values, and adjusts the velocity and
        positions of particles based on inertia, cognitive, and social components.
        It ensures that the updated positions and velocities are clamped within the
        specified bounds.

        The local best positions and fitness values are updated if the current
        fitness is better than the recorded local best. The global best position
        and fitness are determined using helper functions.

        The velocity is updated based on the weighted sum of the previous velocity,
        the cognitive component (personal best), and the social component (global
        best). The population positions are then updated using the new velocities.
        """
        self.gen = self.gen + torch.tensor(1)
        print(self.population.size())
        inner_indices = torch.arange(self.inner_pop_size, device=self.population.device)

        fitness = self.evaluate(self.population[:,inner_indices])
        print(fitness.size())
        compare = self.local_best_fitness > fitness
        self.local_best_location = torch.where(compare[:, None], self.population, self.local_best_location)
        self.local_best_fitness = torch.where(compare, fitness, self.local_best_fitness)
        # self.global_best_location, self.global_best_fitness = min_by(
        #     [self.global_best_location.unsqueeze(0), self.population],
        #     [self.global_best_fitness.unsqueeze(0), fitness],
        # )
        values = torch.cat([self.global_best_location.unsqueeze(1), self.population], dim=1)
        keys = torch.cat([self.global_best_fitness.unsqueeze(1), fitness], dim=1)
        min_index = torch.argmin(keys, dim=1)
        self.global_best_location = values[ :, min_index]
        self.global_best_fitness = keys[ :, min_index]

        rg = torch.rand(self.pop_size, self.inner_pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        rp = torch.rand(self.pop_size, self.inner_pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        velocity = (
            self.w * self.velocity
            + self.phi_p * rp * (self.local_best_location - self.population)
            + self.phi_g * rg * (self.global_best_location - self.population)
        )

        if self.gen<=self.max_gen/2:
            # Add mask for most individuals, except the individuals with the largest norm of (self.global_best_location - self.population)
            distance_vector = self.global_best_location - self.population
            mask = torch.zeros_like(distance_vector, dtype=torch.bool, device=fitness.device)
            global_norm = torch.norm(distance_vector, p=2, dim=2)
            largest_indices = nanmax(global_norm, dim=1).indices.unsqueeze(-1).unsqueeze(-1)
            mask.scatter_(dim=1, index=largest_indices.expand(-1, -1, 3), value=1)
            velocity = torch.where(mask, velocity, torch.zeros_like(velocity, device=fitness.device))
            # debug_print("velocity{}: ",velocity[0])

        population = self.population + velocity
        self.population = clamp(population, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)


def apd_fn(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    obj: torch.Tensor,
    theta: torch.Tensor,
):
    """
    Compute the APD (Angle-Penalized Distance) based on the given inputs.

    :param x: A tensor representing the indices of the partition.
    :param y: A tensor representing the gamma.
    :param z: A tensor representing the angle.
    :param obj: A tensor of shape (n, m) representing the objectives of the solutions.
    :param theta: A tensor representing the parameter theta used for scaling the reference vector.

    :return: A tensor containing the APD values for each solution.
    """
    selected_z = torch.gather(z, 0, torch.relu(x))
    left = (1 + obj.size(1) * theta * selected_z) / y[None, :]
    norm_obj = torch.linalg.vector_norm(obj**2, dim=1)
    right = norm_obj[x]
    return left * right


def new_ref_vec_guided(x: torch.Tensor, f: torch.Tensor, v: torch.Tensor, theta: torch.Tensor):
    """
    Perform the Reference Vector Guided Evolutionary Algorithm (RVEA) selection process.

    This function selects solutions based on the Reference Vector Guided Evolutionary Algorithm.
    It calculates the distances and angles between solutions and reference vectors, and returns
    the next set of solutions to be evolved.

    :param x: A tensor of shape (n, d) representing the current population solutions.
    :param f: A tensor of shape (n, m) representing the objective values for each solution.
    :param v: A tensor of shape (r, m) representing the reference vectors.
    :param theta: A tensor representing the parameter theta used in the APD calculation.

    :return: A tuple containing:
        - next_x: The next selected solutions.
        - next_f: The objective values of the next selected solutions.

    :note:
        The function computes the distances between the solutions and reference vectors,
        and selects the solutions with the minimum APD.
        It currently uses a suboptimal selection implementation, and future improvements
        will optimize the process using a `segment_sort` or `segment_argmin` in CUDA.
    """
    n = f.size(0)
    nv = v.size(0)

    obj = f - nanmin(f, dim=0, keepdim=True)[0]

    obj = maximum(obj, torch.tensor(1e-32, device=f.device))

    cosine = F.cosine_similarity(v.unsqueeze(1), v.unsqueeze(0), dim=-1)

    cosine = torch.where(
        torch.eye(cosine.size(0), dtype=torch.bool, device=f.device),
        0,
        cosine,
    )
    cosine = clamp_float(cosine, 0.0, 1.0)
    gamma = torch.min(torch.acos(cosine), dim=1)[0]

    angle = torch.acos(
        clamp_float(
            F.cosine_similarity(obj.unsqueeze(1), v.unsqueeze(0), dim=-1),
            0.0,
            1.0,
        )
    )

    nan_mask = torch.isnan(obj).any(dim=1)
    associate = torch.argmin(angle, dim=1)
    associate = torch.where(nan_mask, -1, associate)
    associate = associate[:, None]
    partition = torch.arange(0, n, device=f.device)[:, None]
    IndexMatrix = torch.arange(0, nv, device=f.device)[None, :]
    partition = (associate == IndexMatrix) * partition + (associate != IndexMatrix) * -1

    mask = associate != IndexMatrix
    mask_null = mask.sum(dim=0) == n

    apd = apd_fn(partition, gamma, angle, obj, theta)
    apd = torch.where(mask, torch.inf, apd)

    next_ind = torch.argmin(apd, dim=0)
    next_x = torch.where(mask_null.unsqueeze(1), torch.nan, x[next_ind])
    next_f = torch.where(mask_null.unsqueeze(1), torch.nan, f[next_ind])

    rv_performance = torch.ones_like(apd)
    rv_performance = torch.where(mask, 0, rv_performance)
    rv_set_num = torch.sum(rv_performance, dim=0)

    return next_x, next_f, rv_set_num


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
        v0 = v.clone()
        self.pop_size = v.size(0)
        v1 = torch.rand(self.pop_size, self.n_objs, device=device)
        length = ub - lb
        population0 = torch.rand(self.pop_size, self.dim, device=device)
        population0 = length * population0 + lb
        population = torch.cat([population0, torch.full((self.pop_size, self.dim), torch.nan, device=self.device)], dim=0)
        v = torch.cat([v, v1], dim=0)

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size * 2, self.n_objs), device=device).fill_(torch.inf))
        self.reference_vector = Mutable(v)
        self.init_v = v0
        self.ref_vec_init = Parameter(v1, device=device)
        self.gen = Mutable(torch.tensor(0, dtype=int, device=device))
        self.rv_set_num = Mutable(torch.zeros(self.pop_size, self.pop_size * 2, dtype=int, device=device))

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.reference_vector = torch.cat([torch.as_tensor(self.init_v), torch.as_tensor(self.ref_vec_init)], dim=0)
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

    # def _rv_regeneration(self, pop_obj: torch.Tensor, v: torch.Tensor):
    #     """Regenerate reference vectors strategy (PyTorch版本)"""
    #     pop_obj = pop_obj - nanmin(pop_obj, dim=0).values
    #     cosine = F.cosine_similarity(pop_obj.unsqueeze(1), v.unsqueeze(0), dim=-1)
    #     associate = nanmax(cosine, dim=1).indices
    #     invalid = torch.sum((associate.unsqueeze(1) == torch.arange(v.size(0), device=pop_obj.device)), dim=0)
    #     rand = torch.rand((v.size(0), v.size(1)), device=pop_obj.device) * nanmax(pop_obj, dim=0).values
    #     v = torch.where((invalid == 0).unsqueeze(1), rand, v)

    #     return v

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
        mask = torch.where(
            torch.arange(rank.size(0), device=pop.device) < n, torch.tensor(0, dtype=torch.bool, device=pop.device), mask
        )
        mask = mask.unsqueeze(1)

        new_pop = torch.where(mask, pop, torch.nan)
        new_obj = torch.where(mask, obj, torch.nan)

        self.pop = new_pop
        self.fit = new_obj

    def _no_batch_truncation(self, pop: torch.Tensor, obj: torch.Tensor):
        self.pop = pop
        self.fit = obj

    def _update_pop_and_rv(self, survivor: torch.Tensor, survivor_fit: torch.Tensor):
        if self.gen % int(1 / self.fr) == 0:
            v_adapt = self._rv_adaptation(survivor_fit)
        else:
            v_adapt = self._no_rv_adaptation(survivor_fit)
        # v_regen = self._rv_regeneration(survivor_fit, self.reference_vector[self.pop_size :])
        v_regen = self.reference_vector[self.pop_size :]
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

        # v_regen = self._rv_regeneration(survivor_fit, self.reference_vector[self.pop_size :])
        v_regen = self.reference_vector[self.pop_size :]
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

        survivor, survivor_fit, rv_set_num = self.selection(
            merge_pop,
            merge_fit,
            self.reference_vector,
            (self.gen / self.max_gen) ** self.alpha,
        )
        self.rv_set_num = rv_set_num
        self._update_pop_and_rv(survivor, survivor_fit)

    def get_rv_set_num(self):
        return self.rv_set_num, self.pop_size


class solution_transform(torch.nn.Module):
    def __init__(self, n_objs: int):
        super().__init__()
        self.n_objs = n_objs

    def forward(self, x: torch.Tensor):
        y = x.view(x.size(0), -1, self.n_objs)
        y = y / torch.linalg.vector_norm(y, dim=-1, keepdim=True)
        return {"self.algorithm.ref_vec_init": y}


class metric(torch.nn.Module):
    # def __init__(self, pf: torch.Tensor):
    #     super().__init__()
    #     self.pf = pf

    # def forward(self, x: torch.Tensor):
    #     return igd(x, self.pf)
    def __init__(self, rv_set_num, pop_size: int):
        super().__init__()
        self.rv_set_num = rv_set_num
        self.pop_size = pop_size

    def forward(self, x: torch.Tensor):
        rv_mask = torch.where(self.rv_set_num == 0, 1, self.rv_set_num)
        rv_not_average = rv_mask - 1
        rv_not_average_sum = rv_not_average.sum()
        rv_set_sum = self.rv_set_num.sum()
        return rv_not_average_sum - 1 / self.pop_size + 1 / rv_set_sum


class InnerCore(unittest.TestCase):
    def setUp(
        self,
        inner_algo: Algorithm,
        pop_size: int,
        n_objs: int,
        dimensions: int,
        inner_iterations: int,
        num_instances: int,
        num_repeats: int = 1,
    ):
        self.inner_algo = inner_algo(pop_size=pop_size, n_objs=n_objs, lb=torch.zeros(dimensions), ub=torch.ones(dimensions), selection_op=new_ref_vec_guided)
        self.inner_prob = DTLZ7(d=dimensions, m=n_objs)
        self.inner_monitor = HPOFitnessMonitor(multi_obj_metric=metric(self.inner_algo.rv_set_num,self.inner_algo.pop_size))
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
    def setUp(
        self,
        outer_algo: Algorithm,
        num_instances: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        inner_pop_size: int,
        n_objs: int,
        outer_iterations: int,
        hpo_prob: HPOProblemWrapper,
    ):
        if outer_algo == PSO:
            self.outer_algo = PSO(pop_size=num_instances, lb=torch.zeros(lb), ub=torch.ones(ub))
        elif outer_algo == test_meta4.OuterPSO:
            self.outer_algo = outer_algo(
                pop_size=num_instances, lb=torch.zeros(lb), ub=torch.ones(ub), n_objs=n_objs
            )
        elif outer_algo == test_meta5.OuterPSO:
            self.outer_algo = outer_algo(
                pop_size=num_instances, lb=torch.zeros(lb), ub=torch.ones(ub), n_objs=n_objs, max_gen=outer_iterations
            )
        else:
            self.outer_algo = outer_algo(
                pop_size=num_instances, lb=torch.zeros(lb), ub=torch.ones(ub), inner_pop_size=inner_pop_size, n_objs=n_objs, max_gen=outer_iterations
            )
        self.outer_monitor = EvalMonitor(full_sol_history=False)
        self.outer_workflow = StdWorkflow()
        st = solution_transform(n_objs)
        self.outer_workflow.setup(
            self.outer_algo, hpo_prob, monitor=self.outer_monitor, solution_transform=st
        )


if __name__ == "__main__":
    torch.set_default_device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parameters of the inner algorithm
    pop_size = 100
    n_objs = 3
    dimensions = 22

    # Parameters of the hpo problem
    inner_iterations = 100
    num_instances = 1
    num_repeats = 1

    # Iterations of the outer algorithm
    outer_iterations = 100

    inner_algo = [InnerRVEAa]
    outer_algo = [OuterPSO]
    sampling, _ = uniform_sampling(pop_size, n_objs)
    v = sampling.to()
    bound = [v.numel()]
    print(bound)

    for i in range(len(inner_algo)):
        # Initialize the inner core
        inner_core = InnerCore()
        inner_core.setUp(
            inner_algo=inner_algo[i],
            pop_size=pop_size,
            n_objs=n_objs,
            dimensions=dimensions,
            inner_iterations=inner_iterations,
            num_instances=num_instances,
            num_repeats=num_repeats,
        )

        # Initialize the outer core
        outer_core = OuterCore()
        outer_core.setUp(
            outer_algo=outer_algo[i],
            num_instances=num_instances,
            lb=bound[i],
            ub=bound[i],
            inner_pop_size=bound[i],
            n_objs=n_objs,
            outer_iterations=outer_iterations,
            hpo_prob=inner_core.hpo_prob,
        )

        # params = inner_core.hpo_prob.get_init_params()
        # print("init params:\n", params)

        print("Outer algorithm: ", outer_algo[i].__name__)
        print("Inner algorithm: ", inner_algo[i].__name__)
        start_time = time.time()
        for i in range(outer_iterations):
            outer_core.outer_workflow.step()
            if i % 10 == 0:
                print(f"The {i}th iteration and time elapsed: {time.time() - start_time: .4f}(s).")

        print(f"The {outer_iterations}th iteration and time elapsed: {time.time() - start_time: .4f}(s).")
        outer_monitor = outer_core.outer_workflow.get_submodule("monitor")
        print("params:\n", outer_monitor.topk_solutions, "\n")
        print("result:\n", outer_monitor.topk_fitness)
        # print(outer_monitor.best_fitness)

        # params = inner_core.hpo_prob.get_init_params()
