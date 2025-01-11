from typing import Literal

import torch

from ...core import Algorithm, Mutable, Parameter, jit_class
from .adam_step import adam_single_tensor


@jit_class
class ASEBO(Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: torch.Tensor,
        optimizer: Literal["adam"] | None = None,
        lr: float = 0.05,
        lr_decay: float = 1.0,
        lr_limit: float = 0.001,
        sigma: float = 0.03,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        mean_decay: float = 0.0,
        subspace_dims: int | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()

        assert not pop_size & 1

        dim = center_init.shape[0]

        if subspace_dims is None:
            subspace_dims = dim

        # set hyperparameters
        self.lr = Parameter(lr, device=device)
        self.lr_decay = Parameter(lr_decay, device=device)
        self.lr_limit = Parameter(lr_limit, device=device)
        self.sigma_decay = Parameter(sigma_decay, device=device)
        self.sigma_limit = Parameter(sigma_limit, device=device)
        # set value
        self.dim = dim
        self.pop_size = pop_size
        self.optimizer = optimizer
        self.subspace_dims = subspace_dims
        # setup
        center_init.to(device=device)
        self.center = Mutable(center_init)
        self.grad_subspace = Mutable(torch.zeros(self.subspace_dims, self.dim, device=device))
        self.UUT = Mutable(torch.zeros(self.dim, self.dim, device=device))
        self.UUT_ort = Mutable(torch.zeros(self.dim, self.dim, device=device))
        self.sigma = Mutable(torch.tensor(sigma))
        self.alpha = Mutable(torch.tensor(0.1))
        self.gen_counter = Mutable(torch.tensor(0.0))

        if optimizer == "adam":
            self.exp_avg = Mutable(torch.zeros_like(self.center))
            self.exp_avg_sq = Mutable(torch.zeros_like(self.center))
            self.beta1 = Parameter(0.9, device=device)
            self.beta2 = Parameter(0.999, device=device)

    def step(self):
        device = self.center.device

        X = self.grad_subspace
        X = X - torch.mean(X, dim=0)
        U, S, Vt = torch.svd(X, some=True)

        max_abs_cols = torch.argmax(torch.abs(U), dim=0)
        signs = torch.sign(U[max_abs_cols, :])
        U = U * signs
        Vt = Vt * signs

        U = Vt[: int(self.pop_size / 2)]
        UUT = torch.matmul(U.T, U)
        U_ort = Vt[int(self.pop_size / 2) :]
        UUT_ort = torch.matmul(U_ort.T, U_ort)

        UUT = torch.where(self.gen_counter > self.subspace_dims, UUT, torch.zeros(self.dim, self.dim, device=device))

        cov = (
            self.sigma * (self.alpha / self.dim) * torch.eye(self.dim, device=device)
            + ((1 - self.alpha) / int(self.pop_size / 2)) * UUT
        )
        chol = torch.linalg.cholesky(cov)
        noise = torch.randn(self.dim, int(self.pop_size / 2), device=device)

        z_plus = torch.swapaxes(chol @ noise, 0, 1)
        z_plus = z_plus / torch.linalg.norm(z_plus, dim=-1)[:, torch.newaxis]
        z = torch.cat([z_plus, -1.0 * z_plus])

        population = self.center + z

        self.gen_counter = self.gen_counter + 1

        fitness = self.evaluate(population)

        noise = (population - self.center) / self.sigma
        noise_1 = noise[: int(self.pop_size / 2)]
        fit_1 = fitness[: int(self.pop_size / 2)]
        fit_2 = fitness[int(self.pop_size / 2) :]
        fit_diff_noise = noise_1.T @ (fit_1 - fit_2)

        theta_grad = 1.0 / 2.0 * fit_diff_noise
        alpha = torch.linalg.norm(theta_grad @ UUT_ort) / torch.linalg.norm(theta_grad @ self.UUT)

        alpha = torch.where(self.gen_counter > self.subspace_dims, alpha, 1.0)

        self.grad_subspace = torch.cat([self.grad_subspace, theta_grad[torch.newaxis, :]])[1:, :]
        theta_grad /= torch.linalg.norm(theta_grad) / self.dim + 1e-8

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
        sigma = self.sigma * self.sigma_decay
        sigma = torch.maximum(sigma, self.sigma_limit)

        self.center = center
        self.sigma = sigma
        self.alpha = alpha
