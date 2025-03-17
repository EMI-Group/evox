import torch

from evox.core import Algorithm, Mutable, Parameter
from evox.utils import clamp


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
            pop = mean + stdev * torch.randn(self.pop_size, self.dim, device=device)
            pop = clamp(pop, min=self.lb, max=self.ub)
        else:
            pop = torch.rand(self.pop_size, self.dim, device=device)
            pop = pop * (self.ub - self.lb) + self.lb
        velocity = torch.rand(self.pop_size, self.dim, device=device)
        velocity = 2 * length * velocity - length
        # mutable
        self.pop = Mutable(pop)
        self.fit = Mutable(torch.empty(self.pop_size, device=device))
        self.velocity = Mutable(velocity)

    def init_step(self):
        self.fit = self.evaluate(self.pop)

    def step(self):
        """
        Perform a single optimization step using CSO.

        This function updates the position and velocity of each particle in the
        population using the CSO algorithm. The CSO algorithm is an optimization
        algorithm that uses a combination of both the PSO and the DE algorithms to
        search for the optimal solution.
        """
        device = self.pop.device
        # Random pairing and compare their fitness
        left, right = torch.randperm(self.pop_size, device=device).view(2, -1)
        mask = self.fit[left] < self.fit[right]
        # Assign the teachers and students
        teachers = torch.where(mask, left, right)
        students = torch.where(mask, right, left)
        # Calculate the center of the population
        center = torch.mean(self.pop, axis=0)

        # CSO update rule
        lambda1, lambda2, lambda3 = torch.rand((3, self.pop_size // 2, self.dim), device=device)

        student_velocity = (
            lambda1 * self.velocity[students] # Inertia
            + lambda2 * (self.pop[teachers] - self.pop[students]) # Learn from teachers
            + self.phi * lambda3 * (center - self.pop[students]) # converge to the center
        )

        # Clamp the position and velocity to the bounds
        vel_range = self.ub - self.lb
        student_velocity = clamp(student_velocity, -vel_range, vel_range)
        candidates = clamp(self.pop[students] + student_velocity, self.lb, self.ub)
        self.pop[students] = candidates

        # Evaluate the fitness of the new solutions
        candidates_fit = self.evaluate(candidates)
        # Update the population with the new solutions
        self.fit[students] = candidates_fit
