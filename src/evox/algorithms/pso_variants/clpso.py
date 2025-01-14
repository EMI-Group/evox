import torch

from ...core import Algorithm, Mutable, Parameter, jit_class
from ...utils import clamp
from .utils import min_by


@jit_class
class CLPSO(Algorithm):
    """The basic CSO algorithm.

    ## Class Methods

    * `__init__`: Initializes the CLPSO algorithm with given static parameters including lower and upper bounds for particle positions.
    * `setup`: Initializes the CLPSO algorithm and sets up initial population, velocity, and buffers for tracking best local and global positions and fitness values.
    * `step`: Performs a single optimization step using CLPSO, updating local best positions and fitness values, and adjusting velocity and positions based on inertia, cognitive, and social components.

    Note that the `evaluate` method is not defined in this class, it is a proxy function of `Problem.evaluate` set by workflow; therefore, it cannot be used in class methods other than `step`.

    """

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        inertia_weight: float = 0.5,
        const_coefficient: float = 1.5,
        learning_probability: float = 0.05,
        device: torch.device | None = None,
    ):
        """Initialize the CLPSO algorithm with the given static parameters.

        :param pop_size: The size of the population.
        :param lb: The lower bounds of the particle positions. Must be a 1D tensor.
        :param ub: The upper bounds of the particle positions. Must be a 1D tensor.
        :param inertia_weight: The inertia weight (w). Defaults to 0.5.
        :param const_coefficient: The cognitive weight (c). Defaults to 1.5.
        :param learning_probability: The social weight (P_c). Defaults to 0.05.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype
        self.pop_size = pop_size
        self.dim = lb.shape[0]
        self.lb = lb[None, :].to(device=device)
        self.ub = ub[None, :].to(device=device)
        self.w = Parameter(inertia_weight, device=device)
        self.c = Parameter(const_coefficient, device=device)
        self.P_c = Parameter(learning_probability, device=device)
        # get initial value
        length = self.ub - self.lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + self.lb
        velocity = torch.rand(self.pop_size, self.dim, device=device)
        velocity = 2 * length * velocity - length
        # set mutable
        self.population = Mutable(population)
        self.velocity = Mutable(velocity)
        self.personal_best_location = Mutable(population)
        self.personal_best_fitness = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.global_best_location = Mutable(population[0])
        self.global_best_fitness = Mutable(torch.tensor(torch.inf, device=device))

    def step(self):
        """
        Perform a single optimization step using CLPSO.

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
        # evaluate
        fitness = self.evaluate(self.population)
        device = self.population.device
        # Generate random values
        random_coefficient = torch.rand(self.pop_size, self.dim, device=device)
        rand1_index = torch.randint(size=(self.pop_size,), low=0, high=self.pop_size, device=device)
        rand2_index = torch.randint(size=(self.pop_size,), low=0, high=self.pop_size, device=device)
        rand_possibility = torch.rand(self.pop_size, device=device)
        learning_index = torch.where(
            self.personal_best_fitness[rand1_index] < self.personal_best_fitness[rand2_index],
            rand1_index,
            rand2_index,
        )
        # Update personal_best
        compare = self.personal_best_fitness > fitness
        self.personal_best_location = torch.where(compare[:, None], self.population, self.personal_best_location)
        self.personal_best_fitness = torch.where(compare, fitness, self.personal_best_fitness)
        # Update global_best
        self.global_best_location, self.global_best_fitness = min_by(
            [self.global_best_location[None, :], self.population],
            [self.global_best_fitness.unsqueeze(0), fitness],
        )
        # Choose personal_best
        learning_personal_best = self.personal_best_location[learning_index, :]
        personal_best = torch.where(
            (rand_possibility < self.P_c[None])[:, None],
            learning_personal_best,
            self.personal_best_location,
        )
        # Update velocity and position
        velocity = self.w * self.velocity + self.c * random_coefficient * (personal_best - self.population)
        self.velocity = clamp(velocity, self.lb, self.ub)
        population = self.population + velocity
        self.population = clamp(population, self.lb, self.ub)
