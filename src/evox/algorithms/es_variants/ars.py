from typing import Literal

import torch

from ...core import Algorithm, Mutable, Parameter, jit_class
from .adam_step import adam_single_tensor


@jit_class
class ARS(Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: torch.Tensor,
        elite_ratio: float = 0.1,
        lr: float = 0.05,
        sigma: float = 0.03,
        optimizer: Literal["adam"] | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()

        assert not pop_size & 1
        assert 0 <= elite_ratio <= 1

        dim = center_init.shape[0]

        # set hyperparameters
        self.lr = Parameter(lr, device=device)
        self.sigma = Parameter(sigma, device=device)
        # set value
        self.dim = dim
        self.pop_size = pop_size
        self.optimizer = optimizer
        self.elite_pop_size = max(1, int(pop_size / 2 * elite_ratio))
        # setup
        center_init = center_init.to(device=device)
        self.center = Mutable(center_init)

        if optimizer == "adam":
            self.exp_avg = Mutable(torch.zeros_like(self.center))
            self.exp_avg_sq = Mutable(torch.zeros_like(self.center))
            self.beta1 = Parameter(0.9, device=device)
            self.beta2 = Parameter(0.999, device=device)

    def step(self):
        device = self.center.device

        z_plus = torch.randn(int(self.pop_size / 2), self.dim, device=device)
        noise = torch.cat([z_plus, -1.0 * z_plus])
        population = self.center + self.sigma * noise

        fitness = self.evaluate(population)

        noise_1 = noise[: int(self.pop_size / 2)]
        fit_1 = fitness[: int(self.pop_size / 2)]
        fit_2 = fitness[int(self.pop_size / 2) :]
        elite_idx = torch.minimum(fit_1, fit_2).argsort()[: self.elite_pop_size]

        fitness_elite = torch.cat([fit_1[elite_idx], fit_2[elite_idx]])
        sigma_fitness = torch.std(fitness_elite) + 1e-05

        fit_diff = fit_1[elite_idx] - fit_2[elite_idx]
        fit_diff_noise = noise_1[elite_idx].T @ fit_diff

        theta_grad = 1.0 / (self.elite_pop_size * sigma_fitness) * fit_diff_noise

        if self.optimizer is None:
            center = self.center - self.lr * theta_grad
        else:
            center, self.exp_avg, self.exp_avg_sq = adam_single_tensor(
                self.center,
                theta_grad,
                self.exp_avg,
                self.exp_avg_sq,
                self.beta1,
                self.beta2,
                self.lr,
            )
        self.center = center
