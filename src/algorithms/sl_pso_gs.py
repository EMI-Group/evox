import torch
import torch.nn as nn

from ..utils import clamp
from ..core import Parameter, Algorithm, jit_class, trace_impl, batched_random
from .utils import min_by

@jit_class
class SLPSOGS(Algorithm):
    """The basic Particle Swarm Optimization Social Learning PSO Using Gaussian Sampling for Demonstator Choice (SLPSOGS) algorithm.

    ## Class Methods

    * `__init__`: Initializes the SLPSOGS algorithm with given parameters (population size, inertia weight, cognitive weight, and social weight).
    * `setup`: Initializes the SLPSOGS algorithm with given lower and upper bounds for particle positions, and sets up initial population, velocity, and buffers for tracking best local and global positions and fitness values.
    * `step`: Performs a single optimization step using Particle Swarm Optimization (SLPSOGS), updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    """

    def __init__(
            self, 
            pop_size: int, 
            social_influence_factor: float = 0.2,  # epsilon
            demonstrator_choice_factor: float = 0.7,  # theta
        ):
        """
        Initialize the SLPSOGS algorithm with the given parameters.

        Args:
            pop_size (`int`): The size of the population.
            w (`float`, optional): The inertia weight. Defaults to 0.6.
            phi_p (`float`, optional): The cognitive weight. Defaults to 2.5.
            phi_g (`float`, optional): The social weight. Defaults to 0.8.
        """

        super().__init__()
        self.pop_size = pop_size
        # Here, Parameter is used to indicate that these values are hyper-parameters
        # so that they can be correctly traced and vector-mapped
        self.social_influence_factor = Parameter(social_influence_factor)
        self.demonstrator_choice_factor = Parameter(demonstrator_choice_factor)

    def setup(self, lb: torch.Tensor, ub: torch.Tensor):
        """
        Initialize the SLPSOGS algorithm with the given lower and upper bounds.

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
        population = torch.rand(self.pop_size, self.dim)
        population = length * population + lb
        velocity = torch.rand(self.pop_size, self.dim)
        velocity = 2 * length * velocity - length
        # write to self
        self.lb = lb
        self.ub = ub
        # mutable
        self.population = nn.Buffer(population)
        self.velocity = nn.Buffer(velocity)

        # self.local_best_location = nn.Buffer(population)
        # self.local_best_fitness = nn.Buffer(torch.empty(self.pop_size).fill_(torch.inf))
        self.global_best_location = nn.Buffer(population[0])
        self.global_best_fitness = nn.Buffer(torch.tensor(torch.inf))

    def _set_random(self):
        r1 = torch.rand(self.pop_size, self.dim, device = self.population.device)
        r2 = torch.rand(self.pop_size, self.dim, device = self.population.device)
        r3 = torch.rand(self.pop_size, self.dim, device = self.population.device)
        standard_normal_distribution = torch.rand(
            self.pop_size, device=self.population.device
        )
        return r1, r2, r3, standard_normal_distribution

    @trace_impl(_set_random)
    def _trace_set_random(self):
        r1 = batched_random(torch.rand, self.pop_size, self.dim, device = self.population.device)
        r2 = batched_random(torch.rand, self.pop_size, self.dim, device = self.population.device)
        r3 = batched_random(torch.rand, self.pop_size, self.dim, device = self.population.device)
        standard_normal_distribution = batched_random(
            torch.rand, self.pop_size, device=self.population.device
        )
        return r1, r2, r3, standard_normal_distribution

    def step(self):
        """
        Perform a normal optimization step using SLPSOGS.

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

        fitness = self.evaluate(self.population)
        r1, r2, r3, standard_normal_distribution = self._set_random()

        global_best_location, global_best_fitness = min_by(
            [self.global_best_location[None, :], self.population],
            [self.global_best_fitness.unsqueeze(0), fitness],
        )

        # ----------------- Demonstator Choice -----------------
        # sort from largest fitness to smallest fitness (worst to best)
        ranked_population = self.population[torch.argsort(-fitness)]
        sigma = self.demonstrator_choice_factor * (
            self.pop_size - (torch.arange(self.pop_size, device=self.population.device) + 1)
        )
        
        # normal distribution (shape=(self.pop_size,)) means
        # each individual choose a demonstrator by normal distribution
        # with mean = pop_size and std = sigma
        normal_distribution = (
            sigma * (-torch.abs(standard_normal_distribution)) + self.pop_size
        )
        index_k = (
            torch.floor(clamp(normal_distribution, torch.as_tensor(1, device=self.population.device), torch.as_tensor(self.pop_size, device=self.population.device))).long() - 1
        )
        X_k = ranked_population[index_k]
        # ------------------------------------------------------

        X_avg = torch.mean(self.population, dim=0)
        velocity = (
            r1 * self.velocity
            + r2 * (X_k - self.population)
            + r3 * self.social_influence_factor * (X_avg - self.population)
        )
        population = self.population + velocity
        population = clamp(population, self.lb, self.ub)

        self.population = population
        self.velocity = velocity
        self.global_best_location = global_best_location
        self.global_best_fitness = global_best_fitness
        
    # def init_step(self):
    #     """Perform the first step of the SLPSOGS optimization.
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