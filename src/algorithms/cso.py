import torch
from torch import nn

from ..utils import clamp
from ..core import Parameter, Algorithm, jit_class, trace_impl, batched_random


@jit_class
class CSO(Algorithm):
    
    """The basic CSO algorithm.

    ## Class Methods

    * `__init__`: Initializes the CSO algorithm with given parameters.
    * `setup`: Initializes the CSO algorithm with given lower and upper bounds for particle positions, and sets up initial population, velocity, and buffers for tracking best local and global positions and fitness values.
    * `step`: Performs a single optimization step using CSO, updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    

    """
    def __init__(self, pop_size: int, phi: float = 0.0, mean = None,stdev = None):
        """
        Initialize the CSO algorithm with the given parameters.

        Args:
            pop_size (`int`): The size of the population.
            phi (`float`, optional): The inertia weight. Defaults to 0.0.
            mean (`float`, optional): The mean of the normal distribution. Defaults to None.
            stdev (`float`, optional): The standard deviation of the normal distribution. Defaults to None.
        """
        super().__init__()
        self.pop_size = pop_size
        self.phi = Parameter(phi)
        self.mean = mean
        self.stdev =stdev

    def setup(self, lb: torch.Tensor, ub: torch.Tensor):
        """
        Initialize the CSO algorithm with the given lower and upper bounds.

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
            population = torch.clamp(population, min=self.lb, max=self.ub)
        else:
            population = torch.rand(self.pop_size, self.dim)
            population = population * (self.ub - self.lb) + self.lb
        velocity = torch.rand(self.pop_size, self.dim)
        velocity = 2 * length * velocity - length
        # mutable
        self.population = nn.Buffer(population)
        self.velocity = nn.Buffer(velocity)


    def _set_params(self):
        shuffle_idx = torch.randperm(self.pop_size, dtype=torch.int32, device=self.population.device)
        lambda1 = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        lambda2 = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        lambda3 = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        return shuffle_idx, lambda1, lambda2, lambda3

    @trace_impl(_set_params)
    def _trace_set_params(self):
        shuffle_idx = batched_random(torch.randperm, self.pop_size, dtype=torch.int32, device=self.population.device)
        lambda1 = batched_random(torch.rand, self.pop_size // 2, self.dim, device=self.population.device)
        lambda2 = batched_random(torch.rand, self.pop_size // 2, self.dim, device=self.population.device)
        lambda3 = batched_random(torch.rand, self.pop_size // 2, self.dim, device=self.population.device)
        return shuffle_idx, lambda1, lambda2, lambda3

    def step(self):
        """
        Perform a single optimization step using CSO.

        This function updates the position and velocity of each particle in the
        population using the CSO algorithm. The CSO algorithm is an optimization
        algorithm that uses a combination of both the PSO and the DE algorithms to
        search for the optimal solution.
        """
        # Get the shuffled indices, lambda1, lambda2 and lambda3
        shuffle_idx, lambda1, lambda2, lambda3 = self._get_params()

        # Get the population and velocity of the shuffled indices
        pop = self.population[shuffle_idx]
        vec = self.velocity[shuffle_idx]

        # Get the center of the population
        center = torch.mean(self.population, dim=0)[None, :]

        # Evaluate the fitness of the population
        fit = self.evaluate(pop)

        # Split the population into two parts
        left_pop = pop[:self.pop_size//2]
        right_pop = pop[self.pop_size//2:]
        left_vec = vec[:self.pop_size//2]
        right_vec = vec[self.pop_size//2:]
        left_fit = fit[:self.pop_size//2]
        right_fit = fit[self.pop_size//2:]

        # Calculate the mask
        mask = (left_fit < right_fit)[:, None]

        # Update the velocity of the left part of the population
        left_velocity = torch.where(mask, left_vec,
            lambda1 * right_vec
            + lambda2 * (right_pop - left_pop)
            + self.phi * lambda3 * (center - left_pop)
        )

        # Update the velocity of the right part of the population
        right_velocity = torch.where(mask, right_vec,
            lambda1 * left_vec
            + lambda2 * (left_pop - right_pop)
            + self.phi * lambda3 * (center - right_pop)
        )

        # Update the position of the left and right part of the population
        left_pop = left_pop + left_velocity
        right_pop = right_pop + right_velocity

        # Clamp the position and velocity to the bounds
        pop = clamp(torch.cat([left_pop, right_pop]), self.lb, self.ub)
        vec = clamp(torch.cat([left_velocity, right_velocity]), self.lb, self.ub)

        # Update the population and velocity
        self.population = pop
        self.velocity = vec

    # def init_step(self):
    #     """
    #     Perform the first step of CSO.

    #     This function updates the position and velocity of each particle in the
    #     population using the CSO algorithm. The CSO algorithm is an optimization
    #     algorithm that uses a combination of both the PSO and the DE algorithms to
    #     search for the optimal solution.
    #     """
    #     # Get the shuffled indices, lambda1, lambda2 and lambda3
    #     shuffle_idx, lambda1, lambda2, lambda3 = self._get_params()

    #     # Get the population and velocity of the shuffled indices
    #     pop = self.population[shuffle_idx]
    #     vec = self.velocity[shuffle_idx]

    #     # Get the center of the population
    #     center = torch.mean(self.population, dim=0)[None, :]

    #     # Evaluate the fitness of the population
    #     fit = self.evaluate(pop)

    #     # Split the population into two parts
    #     left_pop = pop[:self.pop_size//2]
    #     right_pop = pop[self.pop_size//2:]
    #     left_vec = vec[:self.pop_size//2]
    #     right_vec = vec[self.pop_size//2:]
    #     left_fit = fit[:self.pop_size//2]
    #     right_fit = fit[self.pop_size//2:]

    #     # Calculate the mask
    #     mask = (left_fit < right_fit)[:, None]

    #     # Update the velocity of the left part of the population
    #     left_velocity = torch.where(mask, left_vec,
    #         lambda1 * right_vec
    #         + lambda2 * (right_pop - left_pop)
    #         + self.phi * lambda3 * (center - left_pop)
    #     )

    #     # Update the velocity of the right part of the population
    #     right_velocity = torch.where(mask, right_vec,
    #         lambda1 * left_vec
    #         + lambda2 * (left_pop - right_pop)
    #         + self.phi * lambda3 * (center - right_pop)
    #     )

    #     # Update the position of the left and right part of the population
    #     left_pop = left_pop + left_velocity
    #     right_pop = right_pop + right_velocity

    #     # Clamp the position and velocity to the bounds
    #     pop = clamp(torch.cat([left_pop, right_pop]), self.lb, self.ub)
    #     vec = clamp(torch.cat([left_velocity, right_velocity]), self.lb, self.ub)

    #     # Update the population and velocity
    #     self.population = pop
    #     self.velocity = vec