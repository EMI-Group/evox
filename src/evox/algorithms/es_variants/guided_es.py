from typing import Literal

import torch

from ...core import Algorithm, Mutable, Parameter, jit_class
from .adam_step import adam_single_tensor


@jit_class
class GuidedES(Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: torch.Tensor,
        subspace_dims: int | None = None,
        optimizer: Literal["adam"] | None = None,
        sigma: float = 0.03,
        lr: float = 60,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        device: torch.device | None = None,
    ):
        super().__init__()

        assert pop_size % 2 == 0

        dim = center_init.shape[0]
        if subspace_dims is None:
            subspace_dims = dim

        # set hyperparameters
        self.beta = Parameter(1.0, device=device)
        self.lr = Parameter(lr, device=device)
        self.sigma_decay = Parameter(sigma_decay, device=device)
        self.sigma_limit = Parameter(sigma_limit, device=device)
        # set value
        self.dim = dim
        self.pop_size = pop_size
        self.optimizer = optimizer
        self.subspace_dims = subspace_dims
        # setup
        center_init = center_init.to(device=device)
        self.center = Mutable(center_init)
        self.alpha = Mutable(torch.tensor(0.5))
        self.sigma = Mutable(torch.tensor(sigma))
        self.grad_subspace = Mutable(torch.randn(subspace_dims, dim, device=device))

        if optimizer == "adam":
            self.exp_avg = Mutable(torch.zeros_like(self.center))
            self.exp_avg_sq = Mutable(torch.zeros_like(self.center))
            self.beta1 = Parameter(0.9, device=device)
            self.beta2 = Parameter(0.999, device=device)

    def step(self):
        device = self.center.device

        a = self.sigma * torch.sqrt(self.alpha / self.dim)
        c = self.sigma * torch.sqrt((1.0 - self.alpha) / self.subspace_dims)
        eps_full = torch.randn(self.dim, int(self.pop_size // 2), device=device)

        eps_subspace = torch.randn(self.subspace_dims, int(self.pop_size // 2), device=device)
        Q, _ = torch.linalg.qr(self.grad_subspace)

        z_plus = a * eps_full + c * (Q @ eps_subspace)
        z_plus = torch.swapaxes(z_plus, 0, 1)
        z = torch.cat([z_plus, -1.0 * z_plus])
        population = self.center + z

        fitness = self.evaluate(population)

        noise = z / self.sigma
        noise_1 = noise[: int(self.pop_size / 2)]
        fit_1 = fitness[: int(self.pop_size / 2)]
        fit_2 = fitness[int(self.pop_size / 2) :]
        fit_diff = fit_1 - fit_2
        fit_diff_noise = noise_1.T @ fit_diff
        theta_grad = (self.beta / self.pop_size) * fit_diff_noise

        self.grad_subspace = torch.cat([self.grad_subspace, theta_grad[torch.newaxis, :]])[1:, :]

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

        sigma = torch.maximum(self.sigma_decay * self.sigma, self.sigma_limit)
        self.sigma = sigma
