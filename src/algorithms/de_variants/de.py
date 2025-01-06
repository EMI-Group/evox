from typing import Literal

import torch

from ...core import Parameter, Mutable, Algorithm, jit_class
from ...utils import clamp


@jit_class
class DE(Algorithm):
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

        Args:
            pop_size (`int`): The size of the population.
            lb (`torch.Tensor`): The lower bounds of the particle positions. Must be a 1D tensor.
            ub (`torch.Tensor`): The upper bounds of the particle positions. Must be a 1D tensor.
            base_vector (`Literal["best", "rand"]`, optional): The base vector type. Defaults to "rand".
            num_difference_vectors (`int`, optional): The number of difference vectors in mutation. Defaults to 1.
            differential_weight (`float` or `torch.Tensor`, optional): The differential weight(s), i.e., the factor(s) F of difference vector(s). Defaults to 0.5.
            cross_probability (`float`, optional): The crossover probability CR. Defaults to 0.9.
            batch_size (`int`, optional): The batch size for vectorized non-replace choice. Defaults to 100.
            replace (`bool`, optional): Whether to allow replacement to speed up computation or following the original implementation. Defaults to False.
            mean (`torch.Tensor`, optional): The mean of the normal distribution. Defaults to None.
            stdev (`torch.Tensor`, optional): The standard deviation of the normal distribution. Defaults to None.
            device (`torch.device`, optional): The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        assert pop_size >= 4
        assert cross_probability > 0 and cross_probability <= 1
        assert num_difference_vectors >= 1 and num_difference_vectors < pop_size // 2
        assert base_vector in ["rand", "best"]
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1 and lb.dtype == ub.dtype
        # parameters
        self.pop_size = pop_size
        self.dim = lb.shape[0]
        self.best_vector = True if base_vector == "best" else False
        self.num_difference_vectors = num_difference_vectors
        if num_difference_vectors == 1:
            assert isinstance(differential_weight, float)
        else:
            assert isinstance(
                differential_weight, torch.Tensor
            ) and differential_weight.shape == torch.Size([num_difference_vectors])
        self.differential_weight = Parameter(differential_weight, device=device)
        self.cross_probability = Parameter(cross_probability, device=device)
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
        # mutable
        self.population = Mutable(population)
        self.fitness = Mutable(torch.empty(self.pop_size, device=device).fill_(float("inf")))

    def init_step(self):
        self.fitness = self.evaluate(self.population)
        self.step()

    def step(self):
        device = self.population.device
        num_vec = self.num_difference_vectors * 2 + (0 if self.best_vector else 1)
        random_choices = []
        # Mutation
        # TODO: currently we allow replacement for different vectors, which is not equivalent to the original implementation
        # TODO: we will change to an implementation based on reservoir sampling (e.g., https://github.com/LeviViana/torch_sampling) in the future
        for i in range(num_vec):
            random_choices.append(torch.randperm(self.pop_size, device=device))
        if self.best_vector:
            best_index = torch.argmin(self.fitness)
            base_vector = self.population[best_index][None, :]
            start_index = 0
        else:
            base_vector = self.population[random_choices[0]]
            start_index = 1
        difference_vector = torch.stack(
            [
                self.population[random_choices[i]] - self.population[random_choices[i + 1]]
                for i in range(start_index, num_vec - 1, 2)
            ]
        ).sum(dim=0)
        new_population = base_vector + self.differential_weight * difference_vector
        # Crossover
        cross_prob = torch.rand(self.pop_size, self.dim, device=device)
        random_dim = torch.randint(0, self.dim, (self.pop_size, 1), device=device)
        mask = cross_prob < self.cross_probability
        mask = mask.scatter(dim=1, index=random_dim, value=1)
        new_population = torch.where(mask, new_population, self.population)
        # Selection
        new_population = clamp(new_population, self.lb, self.ub)
        new_fitness = self.evaluate(new_population)
        new_population = torch.where((new_fitness < self.fitness)[:, None], new_population, self.population)
        self.population = new_population
        self.fitness = new_fitness
