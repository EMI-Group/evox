import torch

from ...core import Algorithm, Mutable, Parameter, jit_class
from ...utils import clamp


@jit_class
class CSO(Algorithm):
    """The basic CSO algorithm.

    ## Class Methods

    * `__init__`: Initializes the CSO algorithm with given parameters.
    * `setup`: Initializes the CSO algorithm with given lower and upper bounds for particle positions, and sets up initial population, velocity, and buffers for tracking best local and global positions and fitness values.
    * `step`: Performs a single optimization step using CSO, updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    """

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        phi: float = 0.0,
        mean: torch.Tensor | None = None,
        stdev: torch.Tensor | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the CSO algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param lb: The lower bounds of the particle positions. Must be a 1D tensor.
        :param ub: The upper bounds of the particle positions. Must be a 1D tensor.
        :param phi: The inertia weight. Defaults to 0.0.
        :param mean: The mean of the normal distribution. Defaults to None.
        :param stdev: The standard deviation of the normal distribution. Defaults to None.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype
        self.pop_size = pop_size
        self.dim = lb.shape[0]
        self.phi = Parameter(phi, device=device)
        # setup
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        length = ub - lb
        # write to self
        self.lb = lb
        self.ub = ub
        if mean is not None and stdev is not None:
            population = mean + stdev * torch.randn(self.pop_size, self.dim, device=device)
            population = clamp(population, min=self.lb, max=self.ub)
        else:
            population = torch.rand(self.pop_size, self.dim, device=device)
            population = population * (self.ub - self.lb) + self.lb
        velocity = torch.rand(self.pop_size, self.dim, device=device)
        velocity = 2 * length * velocity - length
        # mutable
        self.population = Mutable(population)
        self.velocity = Mutable(velocity)

    def step(self):
        """
        Perform a single optimization step using CSO.

        This function updates the position and velocity of each particle in the
        population using the CSO algorithm. The CSO algorithm is an optimization
        algorithm that uses a combination of both the PSO and the DE algorithms to
        search for the optimal solution.
        """
        # Get the shuffled indices, lambda1, lambda2 and lambda3
        shuffle_idx = torch.randperm(self.pop_size, device=self.population.device)
        lambda1 = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        lambda2 = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        lambda3 = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        # Get the population and velocity of the shuffled indices
        pop = self.population[shuffle_idx]
        vec = self.velocity[shuffle_idx]
        # Get the center of the population
        center = self.population.mean(dim=0, keepdim=True)
        # Evaluate the fitness of the population
        fit = self.evaluate(pop)
        # Split the population into two parts
        left_pop = pop[: self.pop_size // 2]
        right_pop = pop[self.pop_size // 2 :]
        left_vec = vec[: self.pop_size // 2]
        right_vec = vec[self.pop_size // 2 :]
        left_fit = fit[: self.pop_size // 2]
        right_fit = fit[self.pop_size // 2 :]
        # Calculate the mask
        mask = (left_fit < right_fit)[:, None]
        # Update the velocity of the left part of the population
        left_velocity = torch.where(
            mask,
            left_vec,
            lambda1 * right_vec + lambda2 * (right_pop - left_pop) + self.phi * lambda3 * (center - left_pop),
        )
        # Update the velocity of the right part of the population
        right_velocity = torch.where(
            mask,
            right_vec,
            lambda1 * left_vec + lambda2 * (left_pop - right_pop) + self.phi * lambda3 * (center - right_pop),
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
