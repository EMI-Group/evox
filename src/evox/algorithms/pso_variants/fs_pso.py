import torch

from ...core import Algorithm, Mutable, Parameter, jit_class
from ...utils import clamp
from .utils import min_by


@jit_class
class FSPSO(Algorithm):
    """The Feature Selection PSO algorithm."""

    def __init__(
        self,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        inertia_weight: float = 0.6,  # w
        cognitive_coefficient: float = 2.5,  # c
        social_coefficient: float = 0.8,  # s
        mean=None,
        stdev=None,
        mutate_rate: float = 0.01,  # mutation ratio
        device: torch.device | None = None,
    ):
        """
        Initialize the FSPSO algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param lb: The lower bounds of the particle positions. Must be a 1D tensor.
        :param ub: The upper bounds of the particle positions. Must be a 1D tensor.
        :param inertia_weight: The inertia weight. Defaults to 0.6.
        :param cognitive_coefficient: The cognitive weight. Defaults to 2.5.
        :param social_coefficient: The social weight. Defaults to 0.8.
        :param mean: The mean of the normal distribution. Defaults to None.
        :param stdev: The standard deviation of the normal distribution. Defaults to None.
        :param mutate_rate: The mutation rate. Defaults to 0.01.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        # Here, Parameter is used to indicate that these values are hyper-parameters
        # so that they can be correctly traced and vector-mapped
        self.w = Parameter(inertia_weight, device=device)
        self.phi_p = Parameter(cognitive_coefficient, device=device)
        self.phi_g = Parameter(social_coefficient, device=device)
        self.mutate_rate = Parameter(mutate_rate, device=device)
        # setup
        lb = lb[None, :].to(device=device)
        ub = ub[None, :].to(device=device)
        length = ub - lb
        # write to self
        self.lb = lb
        self.ub = ub
        if mean is not None and stdev is not None:
            population = mean + stdev * torch.randn(self.pop_size, self.dim, device=device)
            population = clamp(population, self.lb, self.ub)
            velocity = stdev * torch.randn(self.pop_size, self.dim, device=device)
        else:
            population = torch.rand(self.pop_size, self.dim, device=device)
            population = population * length + self.lb
            velocity = torch.rand(self.pop_size, self.dim, device=device)
            velocity = velocity * length * 2 - length
        # mutable
        self.population = Mutable(population)
        self.velocity = Mutable(velocity)
        self.local_best_location = Mutable(population)
        self.local_best_fitness = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.global_best_location = Mutable(population[0])
        self.global_best_fitness = Mutable(torch.tensor(torch.inf, device=device))

    def step(self):
        """Perform a normal optimization step using FSPSO."""

        fitness = self.evaluate(self.population)
        device = self.population.device
        # ----------------Enhancement----------------
        ranked_index = torch.argsort(fitness)
        half_pop_size = self.pop_size // 2
        elite_index = ranked_index[:half_pop_size]
        elite_population = self.population[elite_index]
        elite_velocity = self.velocity[elite_index]
        elite_fitness = fitness[elite_index]
        elite_local_best_location = self.local_best_location[elite_index]
        elite_local_best_fitness = self.local_best_fitness[elite_index]

        compare = elite_local_best_fitness > elite_fitness
        local_best_location = torch.where(compare[:, None], elite_population, elite_local_best_location)
        local_best_fitness = torch.where(compare, elite_fitness, elite_local_best_fitness)

        global_best_location, global_best_fitness = min_by(
            [self.global_best_location[None, :], elite_population],
            [self.global_best_fitness.unsqueeze(0), elite_fitness],
        )

        rg = torch.rand(half_pop_size, self.dim, device=device)
        rp = torch.rand(half_pop_size, self.dim, device=device)
        updated_elite_velocity = (
            self.w * elite_velocity
            + self.phi_p * rp * (elite_local_best_location - elite_population)
            + self.phi_g * rg * (global_best_location - elite_population)
        )
        updated_elite_population = elite_population + updated_elite_velocity
        updated_elite_population = clamp(updated_elite_population, self.lb, self.ub)
        updated_elite_velocity = clamp(updated_elite_velocity, self.lb, self.ub)

        # ----------------Crossover----------------
        tournament1 = torch.randint(0, half_pop_size, (half_pop_size,), device=device)
        tournament2 = torch.randint(0, half_pop_size, (half_pop_size,), device=device)
        compare = elite_fitness[tournament1] < elite_fitness[tournament2]
        mutating_pool = torch.where(compare, tournament1, tournament2)

        # Extend (mutate and create new generation)
        original_population = elite_population[mutating_pool]
        offspring_velocity = elite_velocity[mutating_pool]

        offset = (2 * torch.rand(half_pop_size, self.dim, device=device) - 1) * (self.ub - self.lb)
        mutation_prob = torch.rand(half_pop_size, self.dim, device=device)
        mask = mutation_prob < self.mutate_rate
        offspring_population = original_population + torch.where(mask, offset, 0)
        offspring_population = clamp(offspring_population, self.lb, self.ub)
        offspring_local_best_location = offspring_population
        offspring_local_best_fitness = torch.full((half_pop_size,), float("inf"), device=device)

        # Concatenate updated and offspring populations
        population = torch.cat([updated_elite_population, offspring_population], dim=0)
        velocity = torch.cat([updated_elite_velocity, offspring_velocity], dim=0)
        local_best_location = torch.cat([local_best_location, offspring_local_best_location], dim=0)
        local_best_fitness = torch.cat([local_best_fitness, offspring_local_best_fitness], dim=0)
        self.population = population
        self.velocity = velocity
        self.local_best_location = local_best_location
        self.local_best_fitness = local_best_fitness
        self.global_best_location = global_best_location
        self.global_best_fitness = global_best_fitness
