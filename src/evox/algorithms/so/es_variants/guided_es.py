from typing import Literal

import torch

from evox.core import Algorithm, Mutable, Parameter

from .adam_step import adam_single_tensor


class GuidedES(Algorithm):
    """The implementation of the Guided-ES algorithm.

    Reference:
    Guided evolutionary strategies: Augmenting random search with surrogate gradients
    (https://arxiv.org/abs/1806.10230)

    This code has been inspired by or utilizes the algorithmic implementation from evosax.
    More information about evosax can be found at the following URL:
    GitHub Link: https://github.com/RobertTLange/evosax
    """

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
        """Initialize the Guided-ES algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param center_init: The initial center of the population. Must be a 1D tensor.
        :param optimizer: The optimizer to use. Defaults to None. Currently, only "adam" or None is supported.
        :param lr: The learning rate for the optimizer. Defaults to 0.05.
        :param sigma: The standard deviation of the noise. Defaults to 0.03.
        :param sigma_decay: The decay factor for the standard deviation. Defaults to 1.0.
        :param sigma_limit: The minimum value for the standard deviation. Defaults to 0.01.
        :param subspace_dims: The dimension of the subspace. Defaults to None.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        assert pop_size > 1 and pop_size % 2 == 0
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
        self.alpha = Mutable(torch.tensor(0.5, device=device))
        self.sigma = Mutable(torch.tensor(sigma, device=device))
        self.grad_subspace = Mutable(torch.randn(subspace_dims, dim, device=device))

        if optimizer == "adam":
            self.exp_avg = Mutable(torch.zeros_like(self.center))
            self.exp_avg_sq = Mutable(torch.zeros_like(self.center))
            self.beta1 = Parameter(0.9, device=device)
            self.beta2 = Parameter(0.999, device=device)

    def step(self):
        """Run one step of the Guided-ES algorithm.

        The function will sample a population, evaluate their fitness, and then
        update the center and standard deviation of the algorithm using the
        sampled population.
        """
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

        self.grad_subspace = torch.cat([self.grad_subspace, theta_grad[None, :]])[1:, :]

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

    def record_step(self):
        return {"center": self.center, "sigma": self.sigma}
