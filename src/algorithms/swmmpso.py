import torch
import torch.nn as nn

from ..utils import clamp
from ..core import Parameter, Algorithm, jit_class, trace_impl, batched_random
from .topology_utils import (
    build_adjacancy_list_from_matrix, 
    get_neighbour_best_fitness, 
    get_circles_neighbour
    )

from typing import Literal


@jit_class
class SWMMPSO(Algorithm):
    """The basic SWMMPSO algorithm.

    ## Class Methods

    * `__init__`: Initializes the SWMMPSO algorithm with given parameters (population size, max phi_1, max phi_2, max phi, mean, stdev, topology, and shortcut).
    * `setup`: Initializes the SWMMPSO algorithm with given lower and upper bounds for particle positions, and sets up initial population, velocity, and buffers for tracking best local positions and fitness values.
    * `step`: Performs a single optimization step using Particle Swarm Optimization (SWMMPSO), updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    """

    def __init__(
            self, 
            pop_size: int, 
            max_phi_1: float=2.05,
            max_phi_2: float=2.05,
            max_phi: float=4.1,
            mean=None,
            stdev=None,
            topology: Literal["Circles", "Wheels", "Stars", "Random"] = "Circles",
            shortcut: int = 0,
        ):
        """
        Initialize the SWMMPSO algorithm with the given parameters.

        Args:
            pop_size (`int`): The size of the population.
            max_phi_1 (`float`, optional): The maximum phi_1 value. Defaults to 2.05.
            max_phi_2 (`float`, optional): The maximum phi_2 value. Defaults to 2.05.
            max_phi (`float`, optional): The maximum phi value. Defaults to 4.1.
            mean (`float`, optional): The mean of the normal distribution. Defaults to None.
            stdev (`float`, optional): The standard deviation of the normal distribution. Defaults to None.
            topology (`str`, optional): The topology of the swarm. Defaults to "Circles".
            shortcut (`int`, optional): The shortcut value. Defaults to 0.

        Raises:
            `ValueError`: If the topology is not one of the allowed options.
        """

        super().__init__()
        self.pop_size = pop_size
        # Here, Parameter is used to indicate that these values are hyper-parameters
        # so that they can be correctly traced and vector-mapped
        self.max_phi_1 = Parameter(max_phi_1)
        self.max_phi_2 = Parameter(max_phi_2)
        self.max_phi = Parameter(max_phi)
        self.mean = mean
        self.stdev =  stdev
        self.topology = topology
        self.shortcut = Parameter(shortcut)

    def setup(self, lb: torch.Tensor, ub: torch.Tensor):
        """
        Initialize the SWMMPSO algorithm with the given lower and upper bounds.

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

        if self.max_phi > 0:
            phi = torch.ones(self.pop_size, 1) * self.max_phi
        else:
            phi = torch.ones(self.pop_size, 1) * (self.max_phi_1 + self.max_phi_2)

        # equation in original paper where chi = 1-1/phi+\sqrt{\abs{phi**2-4*phi}} result in wrong coefficients
        # --> chi = 1 - 1 / phi + jnp.sqrt(jnp.abs(phi * (phi - 4))) / 2
        # the proper coefficient are chi = 2/(phi - 2 + jnp.sqrt(jnp.abs(phi * (phi - 4))))
        chi = 2 / (phi - 2 + torch.sqrt(torch.abs(phi * (phi - 4))))

        adjacancy_matrix = self._get_topo(population)
        # mutable
        self.population = nn.Buffer(population)
        self.velocity = nn.Buffer(velocity)
        self.local_best_location = nn.Buffer(population)
        self.local_best_fitness = nn.Buffer(torch.empty(self.pop_size).fill_(torch.inf))
        self.neighbor_best_fitness = nn.Buffer(population)
        self.neighbour_best_fitness=nn.Buffer(torch.empty(self.pop_size).fill_(torch.inf))
        self.adjacancy_matrix=adjacancy_matrix
        self.chi=chi
        self.phi=phi
        # self.global_best_location = nn.Buffer(population[0])
        # self.global_best_fitness = nn.Buffer(torch.tensor(torch.inf))

    def _set_random(self):
        phi1 = torch.rand(self.pop_size, self.dim, device=self.population.device) * self.max_phi_1
        phi2 = torch.rand(self.pop_size, self.dim, device=self.population.device) * self.max_phi_2
        return phi1, phi2

    @trace_impl(_set_random)
    def _trace_set_random(self):
        phi1 = batched_random(torch.rand, self.pop_size, self.dim, device=self.population.device) * self.max_phi_1
        phi2 = batched_random(torch.rand, self.pop_size, self.dim, device=self.population.device) * self.max_phi_2
        return phi1, phi2

    def step(self):
        """
        Perform a normal optimization step using SWMMPSO.

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

        # Generate random phi1 and phi2
        phi1 = torch.rand(self.pop_size, self.dim, device=self.population.device) * self.max_phi_1
        phi2 = torch.rand(self.pop_size, self.dim, device=self.population.device) * self.max_phi_2

        # Update local best
        compare = self.local_best_fitness > fitness
        local_best_location = torch.where(
            compare[:, None], self.population, self.local_best_location
        )
        local_best_fitness = torch.minimum(self.local_best_fitness, fitness)

        # Update adjacency matrix and neighborhood best
        adjacancy_matrix = self.adjacancy_matrix
        neighbour_list, _ = build_adjacancy_list_from_matrix(adjacancy_matrix)
        neighbour_best_fitness, neighbour_best_indice = get_neighbour_best_fitness(
            fitness=local_best_fitness, adjacancy_list=neighbour_list
        )
        neighbour_best_location = local_best_location[neighbour_best_indice, :]

        # Update velocity and population
        velocity = self.chi * (
            self.velocity
            + phi1 * (local_best_location - self.population)
            + phi2 * (neighbour_best_location - self.population)
        )
        population = self.population + velocity
        population = clamp(population, self.lb, self.ub)

    def _get_topo(self, population: torch.Tensor):
        """
        Return the adjacency matrix based on the given population and topology.

        Args:
            population (`torch.Tensor`): The current population of particles.

        Returns:
            `torch.Tensor`: The adjacency matrix.
        """

        if self.topology == "Circles":
            adjacancy_matrix = get_circles_neighbour(
                population=population, K=2, shortcut=self.shortcut
            )
        else:
            adjacancy_matrix = torch.zeros(population.size(0), population.size(0), device=population.device)
        return adjacancy_matrix
    
    
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