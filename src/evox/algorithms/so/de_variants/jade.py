import torch

from evox.core import Algorithm, Mutable
from evox.utils import clamp


class JaDE(Algorithm):
    """
    Adaptive Differential Evolution (JaDE) algorithm for optimization.

    ## Class Methods

    * `__init__`: Initializes the JaDE algorithm with the given parameters, including population size, bounds, mutation strategy, and other hyperparameters.
    * `init_step`: Performs the initial evaluation of the population's fitness and proceeds to the first optimization step.
    * `step`: Executes a single optimization step of the JaDE algorithm, involving mutation, crossover, selection, and adaptation of strategy parameters.

    Note that the `evaluate` method is not defined in this class. It is expected to be provided by the `Problem` class or another external component.
    """

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        num_difference_vectors: int = 1,
        mean: torch.Tensor | None = None,
        stdev: torch.Tensor | None = None,
        c: float = 0.1,
        device: torch.device | None = None,
    ):
        """
        Initialize the JaDE algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param lb: The lower bounds of the search space. Must be a 1D tensor.
        :param ub: The upper bounds of the search space. Must be a 1D tensor.
        :param num_difference_vectors: The number of difference vectors used in mutation. Must be at least 1 and less than half of the population size. Defaults to 1.
        :param mean: The mean for initializing the population with a normal distribution. Defaults to None.
        :param stdev: The standard deviation for initializing the population with a normal distribution. Defaults to None.
        :param device: The device to use for tensor computations (e.g., "cpu" or "cuda"). Defaults to None.
        :param c: The learning rate for updating the adaptive parameters F_u and CR_u. Defaults to 0.1.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device

        # Validate input parameters
        assert pop_size >= 4
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype

        # Initialize algorithm parameters
        self.pop_size = pop_size
        self.dim = lb.shape[0]
        self.num_difference_vectors = num_difference_vectors

        # Initialize adaptive parameters F_u and CR_u as scalars
        self.F_u = Mutable(torch.empty(self.pop_size, device=device).fill_(0.5))
        self.CR_u = Mutable(torch.empty(self.pop_size, device=device).fill_(0.5))
        self.c = c

        # Prepare bounds by adding a batch dimension and moving to the specified device
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        self.lb = lb
        self.ub = ub

        # Initialize population
        if mean is not None and stdev is not None:
            # Initialize population using a normal distribution
            population = mean + stdev * torch.randn(self.pop_size, self.dim, device=device)
            population = clamp(population, lb=self.lb, ub=self.ub)
        else:
            # Initialize population uniformly within bounds
            population = torch.rand(self.pop_size, self.dim, device=device)
            population = population * (self.ub - self.lb) + self.lb

        # Mutable attributes to store population and fitness
        self.pop = Mutable(population)
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
        Execute a single optimization step of the JaDE algorithm.

        This involves the following sub-steps:
        1. Mutation: Generate mutant vectors by combining difference vectors and adapting the mutation factor F.
        2. Crossover: Perform crossover between the current population and the mutant vectors based on the crossover probability CR.
        3. Selection: Evaluate the fitness of the new population and select the better individuals between the current and new populations.
        4. Adaptation: Update the adaptive parameters F_u and CR_u based on the successful mutations.
        """
        device = self.pop.device

        # 1) Generate current F_vec and CR_vec with adaptive perturbation
        F_vec = torch.randn(self.pop_size, device=device) * 0.1 + self.F_u
        F_vec = torch.clamp(F_vec, 0.0, 1.0)

        CR_vec = torch.randn(self.pop_size, device=device) * 0.1 + self.CR_u
        CR_vec = torch.clamp(CR_vec, 0.0, 1.0)

        # 2) Mutation: Generate difference vectors and create mutant vectors
        num_vec = self.num_difference_vectors * 2 + 1
        random_choices = []
        for _ in range(num_vec):
            random_choices.append(torch.randint(0, self.pop_size, (self.pop_size,), device=device))

        difference_vectors = torch.stack(
            [self.pop[random_choices[i]] - self.pop[random_choices[i + 1]] for i in range(1, num_vec - 1, 2)]
        ).sum(dim=0)

        pbest_vectors = self._select_rand_pbest_vectors(p=0.05)
        base_vectors_prim = self.pop
        base_vectors_sec = pbest_vectors
        F_vec_2D = F_vec[:, None]

        base_vectors = base_vectors_prim + F_vec_2D * (base_vectors_sec - base_vectors_prim)
        mutation_vectors = base_vectors + difference_vectors * F_vec_2D

        # 3) Crossover: Combine mutant vectors with current population
        cross_prob = torch.rand(self.pop_size, self.dim, device=device)
        random_dim = torch.randint(0, self.dim, (self.pop_size, 1), device=device)
        CR_vec_2D = CR_vec[:, None].expand(-1, self.dim)

        mask = cross_prob < CR_vec_2D
        # Ensure at least one dimension is mutated for each individual
        mask = mask.scatter(dim=1, index=random_dim, value=1)

        new_population = torch.where(mask, mutation_vectors, self.pop)
        new_population = clamp(new_population, self.lb, self.ub)

        # 4) Selection: Choose better individuals based on fitness
        new_fitness = self.evaluate(new_population)
        compare = new_fitness < self.fit  # Boolean mask of successful mutations
        self.pop = torch.where(compare[:, None], new_population, self.pop)
        self.fit = torch.where(compare, new_fitness, self.fit)

        # 5) Adaptation: Update adaptive parameters F_u and CR_u based on successful mutations
        compare_float = compare.float()
        sum_F2 = (F_vec**2 * compare_float).sum()
        sum_F = (F_vec * compare_float).sum()
        sum_CR = (CR_vec * compare_float).sum()
        count = compare_float.sum()

        # Calculate mean_F_success and mean_CR_success, avoiding division by zero
        mean_F_success = torch.where(count > 0, sum_F2 / (sum_F + 1e-9), torch.tensor(0.0, device=device))
        mean_CR_success = torch.where(count > 0, sum_CR / (count + 1e-9), torch.tensor(0.0, device=device))

        # Update adaptive parameters only if there are successful mutations
        updated_F_u = (1 - self.c) * self.F_u + self.c * mean_F_success
        updated_CR_u = (1 - self.c) * self.CR_u + self.c * mean_CR_success

        # Create a boolean mask to check if there are any successful mutations
        count_mask = count > 0.0

        # Update F_u and CR_u based on the mask
        self.F_u = torch.where(count_mask, updated_F_u, self.F_u)
        self.CR_u = torch.where(count_mask, updated_CR_u, self.CR_u)

    def _select_rand_pbest_vectors(self, p: float) -> torch.Tensor:
        """
        Select p-best vectors from the population for mutation.

        :param p: Fraction of the population to consider as top-p best. Must be between 0 and 1.
        :return: A tensor containing selected p-best vectors.
        """
        pop_size = self.pop_size
        top_p_num = max(int(pop_size * p), 1)

        # Sort indices based on fitness in ascending order
        sorted_indices = torch.argsort(self.fit)
        pbest_indices_pool = sorted_indices[:top_p_num]

        # Randomly sample indices from the top-p pool for each individual
        random_indices = torch.randint(0, top_p_num, (self.pop_size,), device=self.pop.device)
        pbest_indices = pbest_indices_pool[random_indices]

        # Retrieve p-best vectors using the sampled indices
        pbest_vectors = self.pop[pbest_indices]

        return pbest_vectors
