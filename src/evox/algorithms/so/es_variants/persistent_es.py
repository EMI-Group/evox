from typing import Literal

import torch

from evox.core import Algorithm, Mutable, Parameter

from .adam_step import adam_single_tensor


class PersistentES(Algorithm):
    """The implementation of the Persistent ES algorithm.

    Reference:
    Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies
    (http://proceedings.mlr.press/v139/vicol21a.html)

    This code has been inspired by or utilizes the algorithmic implementation from evosax.
    More information about evosax can be found at the following URL:
    GitHub Link: https://github.com/RobertTLange/evosax
    """

    def __init__(
        self,
        pop_size: int,
        center_init: torch.Tensor,
        optimizer: Literal["adam"] | None = None,
        lr: float = 0.05,
        sigma: float = 0.03,
        T: int = 100,
        K: int = 10,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        device: torch.device | None = None,
    ):
        """Initialize the Persistent-ES algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param center_init: The initial center of the population. Must be a 1D tensor.
        :param optimizer: The optimizer to use. Defaults to None. Currently, only "adam" or None is supported.
        :param lr: The learning rate for the optimizer. Defaults to 0.05.
        :param sigma: The standard deviation of the noise. Defaults to 0.03.
        :param sigma_decay: The decay factor for the standard deviation. Defaults to 1.0.
        :param sigma_limit: The minimum value for the standard deviation. Defaults to 0.01.
        :param T: The inner problem length. Defaults to 100.
        :param K: The number of inner problems. Defaults to 10.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        assert pop_size > 1 and pop_size % 2 == 0  # Population size must be even
        dim = center_init.shape[0]
        # set hyperparameters
        self.lr = Parameter(lr, device=device)
        self.T = Parameter(T, device=device)
        self.K = Parameter(K, device=device)
        self.sigma_decay = Parameter(sigma_decay, device=device)
        self.sigma_limit = Parameter(sigma_limit, device=device)
        # set value
        self.dim = dim
        self.pop_size = pop_size
        self.optimizer = optimizer
        # setup
        center_init = center_init.to(device=device)
        self.sigma = Mutable(torch.tensor(sigma))
        self.center = Mutable(center_init)
        self.inner_step_counter = Mutable(torch.tensor(0.0))
        self.pert_accum = Mutable(torch.zeros(pop_size, dim, device=device))

        if optimizer == "adam":
            self.exp_avg = Mutable(torch.zeros_like(self.center))
            self.exp_avg_sq = Mutable(torch.zeros_like(self.center))
            self.beta1 = Parameter(0.9, device=device)
            self.beta2 = Parameter(0.999, device=device)

    def step(self):
        device = self.center.device

        pos_perts = torch.randn(self.pop_size // 2, self.dim, device=device) * self.sigma
        neg_perts = -pos_perts
        perts = torch.cat([pos_perts, neg_perts], dim=0)
        pert_accum = self.pert_accum + perts
        population = self.center + perts

        fitness = self.evaluate(population)

        theta_grad = torch.mean(pert_accum * fitness.reshape(-1, 1) / (self.sigma**2), dim=0)

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

        inner_step_counter = self.inner_step_counter + self.K
        self.inner_step_counter = inner_step_counter

        reset = self.inner_step_counter >= self.T
        inner_step_counter = torch.where(reset, 0, inner_step_counter)
        pert_accum = torch.where(reset, torch.zeros(self.pop_size, self.dim, device=device), pert_accum)

        sigma = self.sigma_decay * self.sigma
        sigma = torch.maximum(sigma, self.sigma_limit)

        self.sigma = sigma
        self.pert_accum = pert_accum

    def record_step(self):
        return {"center": self.center, "sigma": self.sigma}
