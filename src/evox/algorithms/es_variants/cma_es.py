import math
import torch
from typing import Tuple

from ...core import Algorithm, Parameter, Mutable
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
        """
        Initialize the CMA-ES algorithm with the given parameters.

        Args:
            pop_size (`int`): The size of the population with the notation $lambda$.
            mean_init (`torch.Tensor`): The initial mean of the population. Must be a 1D tensor.
            sigma (`float`): The standard deviation of the noise.
            device (`torch.device`, optional): The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert sigma > 0, "sigma must be greater than 0."
        assert pop_size > 0, "pop_size must be greater than 0."
        # setvalue
        self.dim = mean_init.size(0)
        self.mean = Mutable(mean_init, device=device)
        self.sigma = Parameter(sigma, device=device)
        if pop_size is None:
            self.pop_size = 4 + math.floor(3 * math.log(mean_init.size(0)))
        else:
            self.pop_size = pop_size
        self.mu = self.pop_size // 2
        mean_init = mean_init.to(device=device)
        self.mean = mean_init 
        if weights is None:
            weights = torch.log((self.pop_size + 1) / 2) - torch.log(torch.arange(1, self.pop_size + 1)).to(device=device)
            self.weights = weights / torch.sum(weights)
        else:
            self.weights = weights.to(device=device)
        self.mu_eff = torch.sum(self.weights) ** 2 / torch.sum(self.weights ** 2)
        self.chi_n = math.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        # set hyperparameters
        # step-size control
        c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c_sigma = Parameter(c_sigma, device=device)
        d_sigma = 1 + 2 * max(math.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1, 0) + self.c_sigma
        self.d_sigma = Parameter(d_sigma, device=device)
        # covariance matrix adaptation
        c_c = (self.mu_eff + 2) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_c = Parameter(c_c, device=device)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_1 = Parameter(c_1, device=device)
        c_mu = min(
            1 - self.c_1,
            (2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((float(self.dim) + 2) ** 2 + self.mu_eff)
            )
        )
        self.c_mu = Parameter(c_mu, device=device)
        # setup
        self.C = Mutable(torch.eye(self.dim, device=device))
        self.p_sigma = Mutable(torch.zeros(self.dim, device=device), device=device)
        self.p_c = Mutable(torch.zeros(self.dim, device=device), device=device)
        
    def step(self):
        device = self.mean.device
        B, D, _ = self._decompsition(self.C)
        y = B @ D @ torch.randn(self.dim, self.pop_size, device=device)
        population = self.mean + self.sigma * y
        fitness = self.evaluate(population)
        fitness, population = sort_by_key(fitness, population)
        old_mean = self.mean
        
        self.mean = self._update_mean(self.mean, population[:, :self.mu])
        self.p_sigma = self._update_path_sigma(self.p_sigma, D, population[:, :self.mu])
        self.sigma = self._update_step_size(self.sigma)
        self.p_c = self._update_path_c(self.p_c, D, population[:, :self.mu], old_mean)
        self.C = self._update_covariance_matrix(self.C, self.p_c, population[:, :self.mu])
    
    def _update_mean(
        self,
        mean: torch.Tensor,
        populations: torch.Tensor
    ) -> torch.Tensor:
        """Update the mean of the CMA-ES algorithm.

        Args:
            mean (`torch.Tensor`): The mean.
            populations (`torch.Tensor`): The selected populations.

        Returns:
            `torch.Tensor`: The updated mean.
        """
        update = mean + self.sigma * self.weights @ (populations - self.mean)
        return update
    
    def _update_covariance_matrix(
        self,
        C: torch.Tensor,
        p_c: torch.Tensor,
        populations: torch.Tensor
    ):
        """Update the covariance matrix of the CMA-ES algorithm.

        Args:
            Cov (`torch.Tensor`): The covariance matrix.
            path (`torch.Tensor`): The path.
            populations (`torch.Tensor`): The populations.
        """
        y = populations - self.mu
        update = (1 - self.c_1 - self.c_mu) * C + self.c_1 * p_c @ p_c + self.c_mu * (y.T * self.weights) @ y
        return update
    
    def _update_path_c(
        self,
        path_c: torch.Tensor,
        populations: torch.Tensor,
        mean: torch.Tensor
    ):
        update = (1 - self.c_c) * path_c + torch.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * self.weights @ (populations - mean)
        return update
    
    def _update_path_sigma(
        self,
        path_sigma: torch.Tensor,
        C_sqrt: torch.Tensor,
        population: torch.Tensor
    ):
        update = (1 - self.c_sigma) * path_sigma + torch.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * C_sqrt @ (population - self.mean) / self.sigma
        return update
    
    def _update_step_size(
        self,
        step_size: torch.Tensor,
    ):
        update = step_size * torch.exp(self.c_sigma / self.d_sigma * (torch.norm(self.p_sigma) / self.chi_n - 1))
        return update
    
    def _decompsition(
        self,
        C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        C = torch.triu(C) + torch.triu(C, diagonal=1).T
        D, B = torch.linalg.eigh(C, eigenvectors=True)
        D = torch.sqrt(D)
        
        return B, D, (B / D) @ B.T
