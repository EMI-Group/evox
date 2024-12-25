import torch
import torch.nn as nn

from ..utils import clamp
from ..core import Parameter, Algorithm, jit_class, trace_impl, batched_random
from .utils import min_by


@jit_class
class FSPSO(Algorithm):
    """The basic Particle Swarm Optimization (PSO) algorithm.

    ## Class Methods

    * `__init__`: Initializes the PSO algorithm with given parameters (population size, inertia weight, cognitive weight, and social weight).
    * `setup`: Initializes the PSO algorithm with given lower and upper bounds for particle positions, and sets up initial population, velocity, and buffers for tracking best local and global positions and fitness values.
    * `step`: Performs a single optimization step using Particle Swarm Optimization (PSO), updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    """

    def __init__(
            self, 
            pop_size: int, 
            inertia_weight: float=0.6,  # w
            cognitive_coefficient: float=2.5,  # c
            social_coefficient: float=0.8,  # s
            mean=None,
            stdev=None,
            mutate_rate: float=0.01,  # mutation ratio
    ):
        """
        Initialize the FSPSO algorithm with the given parameters.

        Args:
            pop_size (`int`): The size of the population.
            inertia_weight (`float`, optional): The inertia weight. Defaults to 0.6.
            cognitive_coefficient (`float`, optional): The cognitive weight. Defaults to 2.5.
            social_coefficient (`float`, optional): The social weight. Defaults to 0.8.
            mean (`float`, optional): The mean of the normal distribution. Defaults to None.
            stdev (`float`, optional): The standard deviation of the normal distribution. Defaults to None. 
            mutate_rate (`float`, optional): The mutation rate. Defaults to 0.01.
        """

        super().__init__()
        self.pop_size = pop_size
        # Here, Parameter is used to indicate that these values are hyper-parameters
        # so that they can be correctly traced and vector-mapped
        self.w = Parameter(inertia_weight)
        self.phi_p = Parameter(cognitive_coefficient)
        self.phi_g = Parameter(social_coefficient)
        self.mean = mean
        self.stdev = stdev
        self.mutate_rate = Parameter(mutate_rate)

    def setup(self, lb: torch.Tensor, ub: torch.Tensor):
        """
        Initialize the FSPSO algorithm with the given lower and upper bounds.

        This function sets up the initial population and velocity for the
        particles within the specified bounds. It also initializes buffers
        for tracking the best local and global positions and fitness values.

        Args:
            lb (`torch.Tensor`): The lower bounds of the particle positions.
                            Must be a 1D tensor.
            ub (`torch.Tensor`): The upper bounds of the particle positions.
                            Must be a 1D tensor.

        Raises:
            `AssertionError`: If the shapes of lb and ub do not match or if
                            they are not 1D tensors.
        """
        assert lb.shape == ub.shape
        assert lb.ndim == 1 and ub.ndim == 1
        self.dim = lb.shape[0]
        # setup
        lb = lb[None, :]
        ub = ub[None, :]
        length = ub - lb
        # write to self
        self.lb = lb
        self.ub = ub

        if self.mean is not None and self.stdev is not None:
            population = self.stdev * torch.randn(self.pop_size, self.dim)
            population = clamp(population, self.lb, self.ub)
            velocity = self.stdev * torch.randn(self.pop_size, self.dim)
        else:
            population = torch.rand(self.pop_size, self.dim)
            population = population * length + self.lb
            velocity = torch.rand(self.pop_size, self.dim)
            velocity = velocity * length * 2 - length
        # mutable
        self.population = nn.Buffer(population)
        self.velocity = nn.Buffer(velocity)
        self.local_best_location = nn.Buffer(population)
        self.local_best_fitness = nn.Buffer(torch.empty(self.pop_size).fill_(torch.inf))
        self.global_best_location = nn.Buffer(population[0])
        self.global_best_fitness = nn.Buffer(torch.tensor(torch.inf))

    def _get_random(self, elite_index: torch.Tensor):
        rg = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        rp = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        tournament1 = torch.randint(0, elite_index.shape[0], (1, self.pop_size - self.pop_size // 2), device=self.population.device)
        tournament2 = torch.randint(0, elite_index.shape[0], (1, self.pop_size - self.pop_size // 2), device=self.population.device)
        return rg, rp, tournament1, tournament2

    @trace_impl(_get_random)
    def _trace_get_random(self, elite_index: torch.Tensor):
        rg = batched_random(torch.rand, self.pop_size // 2, self.dim, device=self.population.device)
        rp = batched_random(torch.rand, self.pop_size // 2, self.dim, device=self.population.device)
        tournament1 = batched_random(torch.randint, 0, elite_index.shape[0], (1, self.pop_size - self.pop_size // 2), device=self.population.device)
        tournament2 = batched_random(torch.randint, 0, elite_index.shape[0], (1, self.pop_size - self.pop_size // 2), device=self.population.device)
        return rg, rp, tournament1, tournament2
    
    def _get_offset_and_mp(self, unmutated_population: torch.Tensor):
        offset = torch.rand(unmutated_population.shape[0], self.dim, device=unmutated_population.device) * (self.ub - self.lb) * 2 - (self.ub - self.lb)
        mp = torch.rand(unmutated_population.shape[0], self.dim, device=unmutated_population.device)
        return offset, mp

    @trace_impl(_get_random)
    def _trace_get_offset_and_mp(self, unmutated_population: torch.Tensor):
        offset = batched_random(torch.rand,unmutated_population.shape[0], self.dim, device=unmutated_population.device) * (self.ub - self.lb) * 2 - (self.ub - self.lb)
        mp = batched_random(torch.rand,unmutated_population.shape[0], self.dim, device=unmutated_population.device)
        return offset, mp
    
    def step(self):
        """
        Perform a normal optimization step using FSPSO.

        This function evaluates the fitness of the current population,
        updates the local and global best positions and fitness values,
        and updates the velocity and positions of the particles.

        The local and global best positions and fitness values are
        updated if the current fitness is better than the recorded
        local and global best. The global best position and fitness
        are determined using helper functions.

        The velocity is updated based on the weighted sum of the previous
        velocity, the cognitive component (personal best), and the social
        component (global best). The population positions are then updated
        using the new velocities.
        """

        fitness = self.evaluate(self.population)

        # ----------------Enhancement----------------
        ranked_index = torch.argsort(fitness)
        elite_index = ranked_index[: self.pop_size // 2]
        # ranked_population = self.population[ranked_index]
        # ranked_velocity = self.velocity[ranked_index]
        elite_population = self.population[elite_index]
        elite_velocity = self.velocity[elite_index]
        elite_fitness = fitness[elite_index]
        elite_local_best_location = self.local_best_location[elite_index]
        elite_local_best_fitness = self.local_best_fitness[elite_index]

        rg, rp, tournament1, tournament2 = self._get_random(elite_index)

        compare = elite_local_best_fitness > elite_fitness
        local_best_location = torch.where(
            compare[:, None], elite_population, elite_local_best_location
        )
        local_best_fitness = torch.minimum(elite_local_best_fitness, elite_fitness)

        # global_best_fitness = torch.cat([self.global_best_fitness, elite_fitness], dim = 0)
        # min_index = torch.argmin(global_best_fitness)   
        # global_best_location =  torch.cat([self.global_best_location[None, :], elite_population], dim = 0)[min_index]
        # global_best_fitness = global_best_fitness[min_index]

        global_best_location, global_best_fitness = min_by(
            [self.global_best_location[None, :], elite_population],
            [self.global_best_fitness.unsqueeze(0), elite_fitness],
        )

        updated_elite_velocity = (
            self.w * elite_velocity
            + self.phi_p * rp * (elite_local_best_location - elite_population)
            + self.phi_g * rg * (global_best_location - elite_population)
        )
        updated_elite_population = elite_population + updated_elite_velocity
        updated_elite_population = clamp(updated_elite_population, self.lb, self.ub)

        # ----------------Crossover----------------
        compare = elite_fitness[tournament1] < elite_fitness[tournament2]
        mutating_pool = torch.where(compare, tournament1, tournament2)

        # Extend (mutate and create new generation)
        unmutated_population = elite_population[mutating_pool.flatten()]
        offspring_velocity = elite_velocity[mutating_pool.flatten()]

        offset, mp = self._get_offset_and_mp(unmutated_population)
        mask = mp < self.mutate_rate
        offspring_population = unmutated_population + torch.where(mask, offset, torch.zeros_like(offset))
        offspring_population = clamp(offspring_population, self.lb, self.ub)
        offspring_local_best_location = offspring_population
        offspring_local_best_fitness = torch.full((offspring_population.shape[0],), float('inf'), device=self.population.device)

        # Concatenate updated and offspring populations
        population = torch.cat((updated_elite_population, offspring_population), dim=0)
        velocity = torch.cat((updated_elite_velocity, offspring_velocity), dim=0)
        local_best_location = torch.cat((local_best_location, offspring_local_best_location), dim=0)
        local_best_fitness = torch.cat((local_best_fitness, offspring_local_best_fitness), dim=0)

        self.population = population
        self.velocity = velocity
        self.local_best_location = local_best_location
        self.local_best_fitness = local_best_fitness
        self.global_best_location = global_best_location
        self.global_best_fitness = global_best_fitness
    
    # def init_step(self):
    #     """Perform the first step of the PSO optimization.
    #     See `step` for more details.
    #     """

    #     fitness = self.evaluate(self.population)
    #     self.local_best_fitness = fitness
    #     self.local_best_location = self.population

    #     rg, _ = self._set_global_and_random(fitness)
    #     velocity = self.w * self.velocity + self.phi_g * rg * (
    #         self.global_best_location - self.population
    #     )
    #     population = self.population + velocity
    #     self.population = clamp(population, self.lb, self.ub)
    #     self.velocity = clamp(velocity, self.lb, self.ub)
