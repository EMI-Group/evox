import torch

from ...core import Algorithm, Mutable, Parameter, jit_class
from ...utils import clamp
from .utils import min_by


@jit_class
class SLPSOGS(Algorithm):
    """The basic Particle Swarm Optimization Social Learning PSO Using Gaussian Sampling for Demonstrator Choice (SLPSOGS) algorithm.

    ## Class Methods

    * `__init__`: Initializes the SLPSOGS algorithm with given parameters (population size, inertia weight, cognitive weight, and social weight).
    * `setup`: Initializes the SLPSOGS algorithm with given lower and upper bounds for particle positions, and sets up initial population, velocity, and buffers for tracking best local and global positions and fitness values.
    * `step`: Performs a single optimization step using Particle Swarm Optimization (SLPSOGS), updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.
    """

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        social_influence_factor: float = 0.2,  # epsilon
        demonstrator_choice_factor: float = 0.7,  # theta
        device: torch.device | None = None,
    ):
        """
        Initialize the SLPSOGS algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param lb: The lower bounds of the particle positions. Must be a 1D tensor.
        :param ub: The upper bounds of the particle positions. Must be a 1D tensor.
        :param w: The inertia weight. Defaults to 0.6.
        :param phi_p: The cognitive weight. Defaults to 2.5.
        :param phi_g: The social weight. Defaults to 0.8.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        # Here, Parameter is used to indicate that these values are hyper-parameters
        # so that they can be correctly traced and vector-mapped
        self.social_influence_factor = Parameter(social_influence_factor, device=device)
        self.demonstrator_choice_factor = Parameter(demonstrator_choice_factor, device=device)
        # setup
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb
        velocity = torch.rand(self.pop_size, self.dim, device=device)
        velocity = 2 * length * velocity - length
        # write to self
        self.lb = lb
        self.ub = ub
        # mutable
        self.population = Mutable(population)
        self.velocity = Mutable(velocity)
        self.global_best_location = Mutable(population[0])
        self.global_best_fitness = Mutable(torch.tensor(torch.inf, device=device))

    def step(self):
        """Perform a normal optimization step using SLPSOGS."""
        device = self.population.device
        fitness = self.evaluate(self.population)
        global_best_location, global_best_fitness = min_by(
            [self.global_best_location.unsqueeze(0), self.population],
            [self.global_best_fitness.unsqueeze(0), fitness],
        )
        # Demonstrator Choice
        # sort from largest fitness to smallest fitness (worst to best)
        ranked_population = self.population[torch.argsort(fitness, descending=True)]
        sigma = self.demonstrator_choice_factor * (self.pop_size - (torch.arange(self.pop_size, device=device) + 1))
        # normal distribution (shape=(self.pop_size,)) means
        # each individual choose a demonstrator by normal distribution
        # with mean = pop_size and std = sigma
        standard_normal_distribution = torch.randn(self.pop_size, device=device)
        normal_distribution = sigma * (-torch.abs(standard_normal_distribution)) + self.pop_size
        index_k = torch.clamp(normal_distribution, 1, self.pop_size).to(dtype=torch.int64) - 1
        X_k = ranked_population[index_k]
        # Update population and velocity
        X_avg = torch.mean(self.population, dim=0)
        r1 = torch.rand(self.pop_size, self.dim, device=device)
        r2 = torch.rand(self.pop_size, self.dim, device=device)
        r3 = torch.rand(self.pop_size, self.dim, device=device)
        velocity = (
            r1 * self.velocity + r2 * (X_k - self.population) + r3 * self.social_influence_factor * (X_avg - self.population)
        )
        population = self.population + velocity
        population = clamp(population, self.lb, self.ub)
        velocity = clamp(velocity, self.lb, self.ub)
        self.population = population
        self.velocity = velocity
        self.global_best_location = global_best_location
        self.global_best_fitness = global_best_fitness
