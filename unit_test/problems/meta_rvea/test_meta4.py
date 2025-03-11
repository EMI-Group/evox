# Outer algorithm: A variant of PSO
# Inner algorithm: RVEAa
# Inner problem: DLTZ7
# Hyperparameters: all instances' random reference vectors
# Strategy: optimize all instances' random reference vectors but only change one reference vector of every instance
# Result of n_objs = 3 (IGD): 0.0464
# Result of n_objs = 5 (IGD): 0.2083
# Result of n_objs = 10 (IGD): 0.8384

# Result of n_objs = 15 (IGD): 1.7881

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from evox.algorithms.pso_variants.utils import min_by
from evox.core import Algorithm, Mutable, Parameter, jit_class, trace_impl
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import non_dominate_rank, ref_vec_guided
from evox.utils import TracingCond, clamp, nanmax, nanmin


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
        n_objs: int,
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
        self.n_objs = n_objs
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
        population = population.view(self.pop_size, -1, self.n_objs)
        population = population / torch.linalg.vector_norm(population, dim=-1, keepdim=True)
        population = population.view(self.pop_size, -1)
        velocity = torch.rand(self.pop_size, self.dim, device=device)
        velocity = 2 * length * velocity - length
        # write to self
        self.lb = lb
        self.ub = ub
        # mutable
        self.pop = Mutable(population, device=device)
        self.velocity = Mutable(velocity, device=device)
        self.local_best_location = Mutable(population)
        self.local_best_fitness = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf), device=device)
        self.global_best_location = Mutable(population[0], device=device)
        self.global_best_fitness = Mutable(torch.tensor(torch.inf, device=device), device=device)
        self.fit = Mutable(torch.empty(self.pop_size, device=device), device=device)

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
        fitness = self.evaluate(self.pop)
        self.fit = fitness
        compare = self.local_best_fitness > fitness
        self.local_best_location = torch.where(compare[:, None], self.pop, self.local_best_location)
        self.local_best_fitness = torch.where(compare, fitness, self.local_best_fitness)
        self.global_best_location, self.global_best_fitness = min_by(
            [self.global_best_location.unsqueeze(0), self.pop],
            [self.global_best_fitness.unsqueeze(0), fitness],
        )
        rg = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        rp = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
        velocity = (
            self.w * self.velocity
            + self.phi_p * rp * (self.local_best_location - self.pop)
            + self.phi_g * rg * (self.global_best_location - self.pop)
        )
        new_population = self.pop + velocity
        new_population = new_population.view(self.pop_size, -1, self.n_objs)
        new_population = new_population / torch.linalg.vector_norm(new_population, dim=-1, keepdim=True)
        velocity = new_population.view(self.pop_size, -1) - self.pop

        # Add mask for most individuals, except the individuals with the largest norm of (self.global_best_location - self.pop)
        distance_instance = self.global_best_location - self.pop
        distance_vector = distance_instance.view(self.pop_size, -1, self.n_objs)
        mask = torch.zeros_like(distance_vector, dtype=torch.bool, device=fitness.device)
        global_norm = torch.norm(distance_vector, p=2, dim=2)
        largest_indices = nanmax(global_norm, dim=1).indices
        mask = mask.index_fill(dim=1, index=largest_indices, value=1)
        mask = mask.view(self.pop_size, -1)
        velocity = torch.where(mask, velocity, 0)

        # Update
        new_population = self.pop + velocity
        self.pop = clamp(new_population, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)

    # def init_step(self):
    #     """Perform the first step of the PSO optimization.

    #     See `step` for more details.
    #     """
    #     fitness = self.evaluate(self.pop)
    #     self.local_best_fitness = fitness
    #     self.local_best_location = self.pop
    #     best_index = torch.argmin(fitness)
    #     self.global_best_location = self.pop[best_index]
    #     self.global_best_fitness = fitness[best_index]
    #     rg = torch.rand(self.pop_size, self.dim, dtype=fitness.dtype, device=fitness.device)
    #     velocity = self.w * self.velocity + self.phi_g * rg * (self.global_best_location - self.pop)
    #     population = self.pop + velocity
    #     self.pop = clamp(population, self.lb, self.ub)
    #     self.velocity = clamp(velocity, self.lb, self.ub)


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

        self.pop = Mutable(population, device=device)
        self.fit = Mutable(torch.empty((self.pop_size * 2, self.n_objs), device=device).fill_(torch.inf), device=device)
        self.reference_vector = Mutable(v, device=device)
        self.init_v = v0
        self.ref_vec_init = Parameter(v1, device=device)
        self.gen = Mutable(torch.tensor(0, dtype=int, device=device), device=device)

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.reference_vector = torch.cat(
            [torch.as_tensor(self.init_v, device=self.device), torch.as_tensor(self.ref_vec_init, device=self.device)], dim=0
        )
        self.fit = self.evaluate(self.pop)

    def _rv_adaptation(self, pop_obj: torch.Tensor):
        max_vals = nanmax(pop_obj, dim=0)[0]
        min_vals = nanmin(pop_obj, dim=0)[0]
        return self.init_v * (max_vals - min_vals)

    def _no_rv_adaptation(self, pop_obj: torch.Tensor):
        return self.reference_vector[: self.pop_size]

    def _mating_pool(self):
        mating_pool = torch.randint(0, self.pop.size(0), (self.pop_size,), device=self.device)
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
        cosine = torch.where(mask, 0, cosine)

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
