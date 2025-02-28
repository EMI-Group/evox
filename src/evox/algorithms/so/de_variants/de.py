from typing import Literal

import torch

from evox.core import Algorithm, Mutable, Parameter
from evox.utils import clamp


class DE(Algorithm):
    """
    Differential Evolution (DE) algorithm for optimization.

    ## Class Methods

    * `__init__`: Initializes the DE algorithm with the given parameters, including population size, bounds, mutation strategy, and other hyperparameters.
    * `init_step`: Performs the initial evaluation of the population's fitness and proceeds to the first optimization step.
    * `step`: Executes a single optimization step of the DE algorithm, involving mutation, crossover, and selection processes.

    Note that the `evaluate` method is not defined in this class. It is expected to be provided by the `Problem` class or another external component.
    """

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        base_vector: Literal["best", "rand"] = "rand",
        num_difference_vectors: int = 1,
        differential_weight: float | torch.Tensor = 0.5,
        cross_probability: float = 0.9,
        mean: torch.Tensor | None = None,
        stdev: torch.Tensor | None = None,
        device: torch.device | None = None,
    ):
        """
        Initialize the DE algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param lb: The lower bounds of the search space. Must be a 1D tensor.
        :param ub: The upper bounds of the search space. Must be a 1D tensor.
        :param base_vector: The base vector type used in mutation. Either "best" or "rand". Defaults to "rand".
        :param num_difference_vectors: The number of difference vectors used in mutation. Must be at least 1 and less than half of the population size. Defaults to 1.
        :param differential_weight: The differential weight(s) (F) applied to difference vectors. Can be a float or a tensor. Defaults to 0.5.
        :param cross_probability: The crossover probability (CR). Defaults to 0.9.
        :param mean: The mean for initializing the population with a normal distribution. Defaults to None.
        :param stdev: The standard deviation for initializing the population with a normal distribution. Defaults to None.
        :param device: The device to use for tensor computations. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device

        # Validate input parameters
        assert pop_size >= 4
        assert 0 < cross_probability <= 1
        assert 1 <= num_difference_vectors < pop_size // 2
        assert base_vector in ["rand", "best"]
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype

        # Initialize parameters
        self.pop_size = pop_size
        self.dim = lb.shape[0]
        self.best_vector = base_vector == "best"
        self.num_difference_vectors = num_difference_vectors

        # Validate and set differential weight
        if num_difference_vectors == 1:
            assert isinstance(differential_weight, float)
        else:
            assert isinstance(differential_weight, torch.Tensor) and differential_weight.shape == torch.Size(
                [num_difference_vectors]
            )
        self.differential_weight = Parameter(differential_weight, device=device)
        self.cross_probability = Parameter(cross_probability, device=device)

        # Move bounds to the specified device and add batch dimension
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        self.lb = lb
        self.ub = ub

        # Initialize population
        if mean is not None and stdev is not None:
            # Initialize population using a normal distribution
            pop = mean + stdev * torch.randn(self.pop_size, self.dim, device=device)
            pop = clamp(pop, lb=self.lb, ub=self.ub)
        else:
            # Initialize population uniformly within bounds
            pop = torch.rand(self.pop_size, self.dim, device=device)
            pop = pop * (self.ub - self.lb) + self.lb

        # Mutable attributes to store population and fitness
        self.pop = Mutable(pop)
        self.fit = Mutable(torch.empty(self.pop_size, device=device).fill_(float("inf")))

    def init_step(self):
        """
        Perform the initial evaluation of the population's fitness and proceed to the first optimization step.

        This method evaluates the fitness of the initial population and then calls the `step` method to perform the first optimization iteration.
        """
        self.fit = self.evaluate(self.pop)
        self.step()

    def step(self):
        """
        Execute a single optimization step of the DE algorithm.

        This involves the following sub-steps:
        1. Mutation: Generate mutant vectors based on the specified base vector strategy (`best` or `rand`) and the number of difference vectors.
        2. Crossover: Perform crossover between the current population and the mutant vectors based on the crossover probability.
        3. Selection: Evaluate the fitness of the new population and select the better individuals between the current and new populations.

        The method ensures that all new population vectors are clamped within the specified bounds.
        """
        device = self.pop.device
        num_vec = self.num_difference_vectors * 2 + (0 if self.best_vector else 1)
        random_choices = []

        # Mutation: Generate random permutations for selecting vectors
        # TODO: Currently allows replacement for different vectors, which is not equivalent to the original implementation
        # TODO: Consider changing to an implementation based on reservoir sampling (e.g., https://github.com/LeviViana/torch_sampling) in the future
        for _ in range(num_vec):
            random_choices.append(torch.randint(0, self.pop_size, (self.pop_size,), device=device))

        # Determine the base vector
        if self.best_vector:
            # Use the best individual as the base vector
            best_index = torch.argmin(self.fit)
            base_vector = self.pop[best_index][None, :]
            start_index = 0
        else:
            # Use randomly selected individuals as base vectors
            base_vector = self.pop[random_choices[0]]
            start_index = 1

        # Generate difference vectors by subtracting randomly chosen population vectors
        difference_vector = torch.stack(
            [self.pop[random_choices[i]] - self.pop[random_choices[i + 1]] for i in range(start_index, num_vec - 1, 2)]
        ).sum(dim=0)

        # Create mutant vectors by adding weighted difference vectors to the base vector
        new_pop = base_vector + self.differential_weight * difference_vector

        # Crossover: Determine which dimensions to crossover based on the crossover probability
        cross_prob = torch.rand(self.pop_size, self.dim, device=device)
        random_dim = torch.randint(0, self.dim, (self.pop_size, 1), device=device)
        mask = cross_prob < self.cross_probability
        mask = mask.scatter(dim=1, index=random_dim, value=1)
        new_pop = torch.where(mask, new_pop, self.pop)

        # Ensure new population is within bounds
        new_pop = clamp(new_pop, self.lb, self.ub)

        # Selection: Evaluate fitness of the new population and select the better individuals
        new_fitness = self.evaluate(new_pop)
        compare = new_fitness < self.fit
        self.pop = torch.where(compare[:, None], new_pop, self.pop)
        self.fit = torch.where(compare, new_fitness, self.fit)
