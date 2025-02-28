import torch

from evox.core import Algorithm, Mutable, Parameter
from evox.utils import clamp

from .utils import min_by


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
            pop = mean + stdev * torch.randn(self.pop_size, self.dim, device=device)
            pop = clamp(pop, self.lb, self.ub)
            velocity = stdev * torch.randn(self.pop_size, self.dim, device=device)
        else:
            pop = torch.rand(self.pop_size, self.dim, device=device)
            pop = pop * length + self.lb
            velocity = torch.rand(self.pop_size, self.dim, device=device)
            velocity = velocity * length * 2 - length
        # mutable
        self.pop = Mutable(pop)
        self.fit = Mutable(torch.empty(self.pop_size, device=device))
        self.velocity = Mutable(velocity)
        self.local_best_location = Mutable(pop)
        self.local_best_fit = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.global_best_location = Mutable(pop[0])
        self.global_best_fit = Mutable(torch.tensor(torch.inf, device=device))

    def init_step(self):
        self.fit = self.evaluate(self.pop)
        self.local_best_fit = self.fit
        self.global_best_fit = torch.min(self.fit)

    def step(self):
        """Perform a normal optimization step using FSPSO."""

        device = self.pop.device
        # ----------------Enhancement----------------
        ranked_index = torch.argsort(self.fit)
        half_pop_size = self.pop_size // 2
        elite_index = ranked_index[:half_pop_size]
        elite_pop = self.pop[elite_index]
        elite_velocity = self.velocity[elite_index]
        elite_fit = self.fit[elite_index]
        elite_local_best_location = self.local_best_location[elite_index]
        elite_local_best_fit = self.local_best_fit[elite_index]

        compare = elite_local_best_fit > elite_fit
        local_best_location = torch.where(compare[:, None], elite_pop, elite_local_best_location)
        local_best_fit = torch.where(compare, elite_fit, elite_local_best_fit)

        global_best_location, global_best_fit = min_by(
            [self.global_best_location[None, :], elite_pop],
            [self.global_best_fit.unsqueeze(0), elite_fit],
        )

        rg = torch.rand(half_pop_size, self.dim, device=device)
        rp = torch.rand(half_pop_size, self.dim, device=device)
        updated_elite_velocity = (
            self.w * elite_velocity
            + self.phi_p * rp * (elite_local_best_location - elite_pop)
            + self.phi_g * rg * (global_best_location - elite_pop)
        )
        updated_elite_pop = elite_pop + updated_elite_velocity
        updated_elite_pop = clamp(updated_elite_pop, self.lb, self.ub)
        updated_elite_velocity = clamp(updated_elite_velocity, self.lb, self.ub)

        # ----------------Crossover----------------
        tournament1 = torch.randint(0, half_pop_size, (half_pop_size,), device=device)
        tournament2 = torch.randint(0, half_pop_size, (half_pop_size,), device=device)
        compare = elite_fit[tournament1] < elite_fit[tournament2]
        mutating_pool = torch.where(compare, tournament1, tournament2)

        # Extend (mutate and create new generation)
        original_population = elite_pop[mutating_pool]
        offspring_velocity = elite_velocity[mutating_pool]

        offset = (2 * torch.rand(half_pop_size, self.dim, device=device) - 1) * (self.ub - self.lb)
        mutation_prob = torch.rand(half_pop_size, self.dim, device=device)
        mask = mutation_prob < self.mutate_rate
        offspring_population = original_population + torch.where(mask, offset, 0)
        offspring_population = clamp(offspring_population, self.lb, self.ub)
        offspring_local_best_location = offspring_population
        offspring_local_best_fit = torch.full((half_pop_size,), float("inf"), device=device)

        # Concatenate updated and offspring populations
        pop = torch.cat([updated_elite_pop, offspring_population], dim=0)
        velocity = torch.cat([updated_elite_velocity, offspring_velocity], dim=0)
        local_best_location = torch.cat([local_best_location, offspring_local_best_location], dim=0)
        local_best_fit = torch.cat([local_best_fit, offspring_local_best_fit], dim=0)
        self.pop = pop
        self.velocity = velocity
        self.local_best_location = local_best_location
        self.local_best_fit = local_best_fit
        self.global_best_location = global_best_location
        self.global_best_fit = global_best_fit

        self.fit = self.evaluate(self.pop)
