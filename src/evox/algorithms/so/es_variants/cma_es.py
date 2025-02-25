import math
from typing import Tuple

import torch

from evox.core import Algorithm, Mutable, Parameter

from .sort_utils import sort_by_key


class CMAES(Algorithm):
    """The CMA-ES algorithm as described in "The CMA Evolution Strategy: A Tutorial" from https://arxiv.org/abs/1604.00772."""

    def __init__(
        self,
        mean_init: torch.Tensor,
        sigma: float,
        pop_size: int | None = None,
        weights: torch.Tensor | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the CMA-ES algorithm with the given parameters.

        :param pop_size: The size of the population with the notation $\\lambda$.
        :param mean_init: The initial mean of the population. Must be a 1D tensor.
        :param sigma: The overall standard deviation, i.e., the step size of the algorithm.
        :param weights: The recombination weights of the population. Defaults to None and is calculated automatically with recommended values.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert sigma > 0, "sigma must be greater than 0."

        # setvalue
        self.dim = mean_init.size(0)
        self.mean = Mutable(mean_init, device=device)
        self.sigma = Mutable(torch.tensor(sigma, device=device))
        if pop_size is None:
            self.pop_size = 4 + math.floor(3 * math.log(mean_init.size(0)))
        else:
            self.pop_size = pop_size
        assert self.pop_size > 0, "pop_size must be greater than 0."
        self.mu = self.pop_size // 2
        mean_init = mean_init.to(device=device)
        self.mean = mean_init.unsqueeze(0)

        if weights is None:
            pop_size = torch.tensor(self.pop_size, device=device)
            weights = torch.log((pop_size + 1) / 2) - torch.log(torch.arange(1, self.mu + 1)).to(device=device)
            self.weights = weights / torch.sum(weights)
        else:
            self.weights = weights.to(device=device)
        self.weights = self.weights.unsqueeze(0)

        self.mu_eff = torch.sum(self.weights) ** 2 / torch.sum(self.weights**2)
        mu_eff = float(self.mu_eff)
        self.chi_n = math.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))

        # set hyperparameters
        # step-size control
        c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c_sigma = Parameter(c_sigma, device=device)
        d_sigma = 1 + 2 * max(math.sqrt((mu_eff - 1) / (self.dim + 1)) - 1, 0) + self.c_sigma
        self.d_sigma = Parameter(d_sigma, device=device)

        # covariance matrix adaptation
        c_c = (mu_eff + 2) / (self.dim + 4 + 2 * mu_eff / self.dim)
        self.c_c = Parameter(c_c, device=device)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_1 = Parameter(c_1, device=device)
        c_mu = min(1 - self.c_1, (2 * (self.mu_eff - 2 + 1 / mu_eff) / ((float(self.dim) + 2) ** 2 + mu_eff)))
        self.c_mu = Parameter(c_mu, device=device)

        # decomposition iter
        decomp_per_iter = 1 / (c_1 + c_mu) / self.dim / 10
        decomp_per_iter = max(math.floor(decomp_per_iter), 1)
        self.decomp_per_iter = Parameter(decomp_per_iter, device=device)

        # setup
        self.iteration = Mutable(torch.tensor(0, device=device))
        self.C = Mutable(torch.eye(self.dim, device=device))
        self.C_invsqrt = Mutable(torch.eye(self.dim, device=device))
        self.B = Mutable(torch.eye(self.dim, device=device))
        self.D = Mutable(torch.eye(self.dim, device=device))

        self.p_sigma = Mutable(torch.zeros(self.dim, device=device))
        self.p_c = Mutable(torch.zeros(self.dim, device=device))

    def step(self):
        """The main step of the CMA-ES algorithm.

        In this step, the algorithm generates a new population, evaluates it, and updates the mean, covariance matrix, and step size correspondingly.
        """
        self.iteration = self.iteration + 1

        y = torch.randn(self.pop_size, self.dim, device=self.mean.device) @ self.D @ self.B
        population = self.mean + self.sigma * y

        fitness = self.evaluate(population)
        fitness, population = sort_by_key(fitness, population)
        population_selected = population[: self.mu]
        new_mean = self._update_mean(self.mean, population_selected)
        delta_mean = (new_mean - self.mean).squeeze(0)

        self.p_sigma = self._update_path_sigma(self.p_sigma, self.C_invsqrt, delta_mean)
        h_sigma = (
            torch.norm(self.p_sigma) / torch.sqrt(1 - (1 - self.c_sigma) ** (2 * self.iteration))
            < (1.4 + 2 / (self.dim + 1)) * self.chi_n
        ).float()

        self.p_c = self._update_path_c(self.p_c, h_sigma, delta_mean)
        self.C = self._update_covariance_matrix(self.C, self.p_c, population_selected, self.mean, h_sigma)
        self.sigma = self._update_step_size(self.sigma)
        self.mean = new_mean

        self.B, self.D, self.C_invsqrt = self._conditional_decomposition(self.iteration, self.C)

    def _update_mean(self, mean: torch.Tensor, population: torch.Tensor) -> torch.Tensor:
        update = self.weights @ (population - mean)
        return mean + update

    def _update_covariance_matrix(
        self, C: torch.Tensor, p_c: torch.Tensor, population: torch.Tensor, old_mean: torch.Tensor, h_sigma
    ):
        y = (population - old_mean) / self.sigma
        update = (
            (1 - self.c_1 - self.c_mu) * C
            + self.c_1 * (p_c @ p_c.T + (1 - h_sigma) * self.c_c * (2 - self.c_c) * C)
            + self.c_mu * (y.T * self.weights) @ y
        )
        return update

    def _update_path_c(self, path_c: torch.Tensor, h_sigma, delta_mean: torch.Tensor):
        update = (1 - self.c_c) * path_c + h_sigma * torch.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff
        ) * delta_mean / self.sigma
        return update

    def _update_path_sigma(self, path_sigma: torch.Tensor, C_invsqrt: torch.Tensor, delta_mean: torch.Tensor):
        update = (1 - self.c_sigma) * path_sigma + torch.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff
        ) * C_invsqrt @ delta_mean / self.sigma
        return update

    def _update_step_size(
        self,
        step_size: torch.Tensor,
    ):
        update = step_size * torch.exp(self.c_sigma / self.d_sigma * (torch.norm(self.p_sigma) / self.chi_n - 1))
        return update

    def _conditional_decomposition(self, iteration: torch.Tensor, C: torch.Tensor):
        # limitation of torch.cond: https://pytorch.org/docs/stable/generated/torch.cond.html
        # It currently does not support pytree outputs, so we need to stack the outputs into a single tensor.
        B, D, C_invsqrt = torch.cond(
            iteration % self.decomp_per_iter == 0,
            self._decomposition,
            self._no_decomposition,
            (C,),
        )
        return B, D, C_invsqrt

    def _no_decomposition(self, C: torch.Tensor):
        return torch.stack([self.B, self.D, self.C_invsqrt], dim=0)

    def _decomposition(
        self,
        C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        C = (C + C.T) / 2
        D, B = torch.linalg.eigh(C)
        D = torch.clamp(D, min=1e-8)
        C_invsqrt = B @ torch.diag(1.0 / torch.sqrt(D)) @ B.T
        D = torch.diag(D)
        D = torch.sqrt(D)
        D = B @ D
        return torch.stack([B.T, D, C_invsqrt], dim=0)

    def record_step(self):
        return {
            "mean": self.mean,
            "sigma": self.sigma,
        }
