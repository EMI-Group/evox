# Outer algorithm: A variant of PSO
# Inner algorithm: RVEAa
# Inner problem: DLTZ7
# Hyperparameters: all instances' random reference vectors
# Strategy: optimize all instances' random reference vectors

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from evox.algorithms.so.pso_variants.utils import min_by
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import non_dominate_rank, ref_vec_guided
from evox.utils import clamp, nanmax, nanmin


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

        new_population = self.pop + velocity
        self.pop = clamp(new_population, self.lb, self.ub)
        self.velocity = clamp(velocity, self.lb, self.ub)


class InnerRVEAa(Algorithm):
    """
    An implementation of the Reference Vector Guided Evolutionary Algorithm embedded with the reference vector regeneration strategy (RVEAa) for multi-objective optimization problems.

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
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)

        self.alpha = alpha
        self.fr = torch.as_tensor(fr)
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
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb
        v1 = torch.rand(self.pop_size, self.n_objs, device=device)
        v = torch.cat([v, v1], dim=0)

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        self.reference_vector = Mutable(v)
        self.init_v = v0
        self.ref_vec_init = Parameter(v1, device=device)
        self.gen = Mutable(torch.tensor(0, dtype=int, device=device))

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop)
        self.reference_vector = torch.cat(
            [torch.as_tensor(self.init_v, device=self.device), torch.as_tensor(self.ref_vec_init, device=self.device)], dim=0
        )

    def _rv_adaptation(self, pop_obj: torch.Tensor):
        max_vals = nanmax(pop_obj, dim=0)[0]
        min_vals = nanmin(pop_obj, dim=0)[0]
        return self.init_v * (max_vals - min_vals)

    def _no_rv_adaptation(self, pop_obj: torch.Tensor):
        return self.reference_vector[: self.pop_size].clone()

    def _mating_pool(self):
        valid_mask = ~torch.isnan(self.pop).all(dim=1)
        num_valid = torch.sum(valid_mask, dtype=torch.int32)
        mating_pool = torch.randint(0, torch.iinfo(torch.int32).max, (self.pop_size,), device=self.pop.device) % num_valid
        sorted_indices = torch.where(valid_mask, torch.arange(self.pop.size(0), device=self.device), torch.inf)
        sorted_indices = torch.argsort(sorted_indices, stable=True)
        pop = self.pop[sorted_indices[mating_pool]]
        return pop

    def _rv_regeneration(self, pop_obj: torch.Tensor, v: torch.Tensor):
        pop_obj = pop_obj - nanmin(pop_obj, dim=0).values
        cosine = F.cosine_similarity(pop_obj.unsqueeze(1), v.unsqueeze(0), dim=-1)

        mask = torch.isnan(cosine)
        input_tensor = torch.where(mask, -torch.inf, cosine)
        associate = input_tensor.max(dim=1, keepdim=False).indices
        associate = torch.where(input_tensor[:, 0] == -torch.inf, -1, associate)

        invalid = torch.sum((associate.unsqueeze(1) == torch.arange(v.size(0), device=pop_obj.device)), dim=0)
        rand = torch.rand((v.size(0), v.size(1)), device=pop_obj.device) * nanmax(pop_obj, dim=0).values
        new_v = torch.where((invalid == 0).unsqueeze(1), rand, v)

        return new_v

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
        mask = torch.where(
            torch.arange(rank.size(0), device=pop.device) < n, torch.tensor(0, dtype=torch.bool, device=pop.device), mask
        )
        mask = mask.unsqueeze(1)

        new_pop = torch.where(mask, pop, torch.nan)
        new_obj = torch.where(mask, obj, torch.nan)

        return new_pop, new_obj

    def _no_batch_truncation(self, pop: torch.Tensor, obj: torch.Tensor):
        return pop.clone(), obj.clone()

    def _update_pop_and_rv(self, survivor: torch.Tensor, survivor_fit: torch.Tensor):
        v_regen = self._rv_regeneration(survivor_fit, self.reference_vector[self.pop_size :])
        if torch.compiler.is_compiling():
            v_adapt = torch.cond(
                self.gen % (1 / self.fr).to(dtype=torch.int) == 0, self._rv_adaptation, self._no_rv_adaptation, (survivor_fit,)
            )
            self.pop, self.fit = torch.cond(
                self.gen == self.max_gen, self._batch_truncation, self._no_batch_truncation, (survivor, survivor_fit)
            )
        else:
            print(torch.compiler.is_compiling())
            if self.gen % (1 / self.fr).to(dtype=torch.int) == 0:
                v_adapt = self._rv_adaptation(survivor_fit)
            else:
                v_adapt = self._no_rv_adaptation(survivor_fit)
            if self.gen == self.max_gen:
                self.pop, self.fit = self._batch_truncation(survivor, survivor_fit)
            else:
                self.pop, self.fit = self._no_batch_truncation(survivor, survivor_fit)
        self.reference_vector = torch.cat([v_adapt, v_regen], dim=0)

    def step(self):
        """Perform a single optimization step."""

        self.gen = self.gen + 1
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
