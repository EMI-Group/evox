from typing import Literal

import torch

from evox.core import Algorithm, Mutable, Parameter

from .adam_step import adam_single_tensor


class ESMC(Algorithm):
    """The implementation of the DES algorithm.

    Reference:
    Learn2Hop: Learned Optimization on Rough Landscapes
    (https://proceedings.mlr.press/v139/merchant21a.html)

    This code has been inspired by or utilizes the algorithmic implementation from evosax.
    More information about evosax can be found at the following URL:
    GitHub Link: https://github.com/RobertTLange/evosax
    """

    def __init__(
        self,
        pop_size: int,
        center_init: torch.Tensor,
        optimizer: Literal["adam"] | None = None,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        lr: float = 0.05,
        sigma: float = 0.03,
        device: torch.device | None = None,
    ):
        """Initialize the ESMC algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param center_init: The initial center of the population. Must be a 1D tensor.
        :param elite_ratio: The ratio of elite population. Defaults to 0.1.
        :param lr: The learning rate for the optimizer. Defaults to 0.05.
        :param sigma_decay: The decay factor for the standard deviation. Defaults to 1.0.
        :param sigma_limit: The minimum value for the standard deviation. Defaults to 0.01.
        :param optimizer: The optimizer to use. Defaults to None. Currently, only "adam" or None is supported.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        assert pop_size > 1
        dim = center_init.shape[0]
        # set hyperparameters
        self.lr = Parameter(lr, device=device)
        self.sigma_decay = Parameter(sigma_decay, device=device)
        self.sigma_limit = Parameter(sigma_limit, device=device)
        # set value
        self.dim = dim
        self.pop_size = pop_size
        self.optimizer = optimizer
        # setup
        center_init = center_init.to(device=device)
        self.center = Mutable(center_init)
        self.sigma = Mutable(torch.ones(self.dim, device=device) * sigma)

        if optimizer == "adam":
            self.exp_avg = Mutable(torch.zeros_like(self.center))
            self.exp_avg_sq = Mutable(torch.zeros_like(self.center))
            self.beta1 = Parameter(0.9, device=device)
            self.beta2 = Parameter(0.999, device=device)

    def step(self):
        """One iteration of the ESMC algorithm.

        This function will sample a population, evaluate their fitness, and then
        update the center and standard deviation of the algorithm using the
        sampled population.
        """
        device = self.center.device

        z_plus = torch.randn(int(self.pop_size / 2), self.dim, device=device)
        z = torch.cat([torch.zeros(1, self.dim, device=device), z_plus, -1.0 * z_plus])

        population = self.center + z * self.sigma.reshape(1, self.dim)

        fitness = self.evaluate(population)

        noise = (population - self.center) / self.sigma
        bline_fitness = fitness[0]
        noise = noise[1:]
        fitness = fitness[1:]
        noise_1 = noise[: int((self.pop_size - 1) / 2)]
        fit_1 = fitness[: int((self.pop_size - 1) / 2)]
        fit_2 = fitness[int((self.pop_size - 1) / 2) :]
        fit_diff = torch.minimum(fit_1, bline_fitness) - torch.minimum(fit_2, bline_fitness)
        fit_diff_noise = noise_1.T @ fit_diff

        theta_grad = 1.0 / int((self.pop_size - 1) / 2) * fit_diff_noise

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

        sigma = torch.maximum(self.sigma * self.sigma_decay, self.sigma_limit)
        self.sigma = sigma

    def record_step(self):
        return {"center": self.center, "sigma": self.sigma}
