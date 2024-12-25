import torch
from torch import nn

from ..utils import clamp
from ..core import Parameter, Algorithm, jit_class, trace_impl, batched_random


@jit_class
class CSO(Algorithm):
    def __init__(self, pop_size: int, phi: float = 0.0, mean = None,stdev = None):
        super().__init__()
        self.pop_size = pop_size
        self.phi = Parameter(phi)
        self.mean = mean
        self.stdev =stdev

    def setup(self, lb: torch.Tensor, ub: torch.Tensor):
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


    def _get_params(self):
        shuffle_idx = torch.randperm(self.pop_size, dtype=torch.int32, device=self.population.device)
        lambda1 = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        lambda2 = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        lambda3 = torch.rand(self.pop_size // 2, self.dim, device=self.population.device)
        return shuffle_idx, lambda1, lambda2, lambda3

    @trace_impl(_get_params)
    def _trace_get_params(self):
        shuffle_idx = batched_random(torch.randperm, self.pop_size, dtype=torch.int32, device=self.population.device)
        lambda1 = batched_random(torch.rand, self.pop_size // 2, self.dim, device=self.population.device)
        lambda2 = batched_random(torch.rand, self.pop_size // 2, self.dim, device=self.population.device)
        lambda3 = batched_random(torch.rand, self.pop_size // 2, self.dim, device=self.population.device)
        return shuffle_idx, lambda1, lambda2, lambda3

    def step(self):
        shuffle_idx, lambda1, lambda2, lambda3 = self._get_params()
        # inv_shuffle_idx = torch.argsort(shuffle_idx)
        pop = self.population[shuffle_idx]
        vec = self.velocity[shuffle_idx]
        center = torch.mean(self.population, dim=0)[None, :]
        fit = self.evaluate(pop)
        left_pop = pop[:self.pop_size//2]
        right_pop = pop[self.pop_size//2:]
        left_vec = vec[:self.pop_size//2]
        right_vec = vec[self.pop_size//2:]
        left_fit = fit[:self.pop_size//2]
        right_fit = fit[self.pop_size//2:]
        mask = (left_fit < right_fit)[:, None]
        
        left_velocity = torch.where(mask, left_vec, 
            lambda1 * right_vec
            + lambda2 * (right_pop - left_pop)
            + self.phi * lambda3 * (center - left_pop)
        )
        right_velocity = torch.where(mask, right_vec, 
            lambda1 * left_vec
            + lambda2 * (left_pop - right_pop)
            + self.phi * lambda3 * (center - right_pop)
        )
        
        left_pop = left_pop + left_velocity
        right_pop = right_pop + right_velocity
        pop = clamp(torch.cat([left_pop, right_pop]), self.lb, self.ub)
        vec = clamp(torch.cat([left_velocity, right_velocity]), self.lb, self.ub)

        self.population = pop
        self.velocity = vec

    # def init_step(self):
    #     """Perform the first step of the PSO optimization.
    #     See `step` for more details.
    #     """

    #     fitness = self.evaluate(self.population)
    #     self.local_best_fitness = fitness
    #     self.local_best_location = self.population

    #     rg, _ = self._set_global_and_random(fitness)
    #     velocity = self.w * self.velocity + self.phi_g * rg * (
    #         self.global_best_location - self.population
    #     )
    #     population = self.population + velocity
    #     self.population = clamp(population, self.lb, self.ub)
    #     self.velocity = clamp(velocity, self.lb, self.ub)