import math
from typing import Literal

import torch
import torch.nn.functional as F

from evox.core import Algorithm, Mutable, Parameter


class SNES(Algorithm):
    """The implementation of the SNES algorithm.

    Reference:
    Natural Evolution Strategies
    (https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf)

    This code has been inspired by or utilizes the algorithmic implementation from evosax.
    More information about evosax can be found at the following URL:
    GitHub Link: https://github.com/RobertTLange/evosax
    """

    def __init__(
        self,
        pop_size: int,
        center_init: torch.Tensor,
        sigma: float = 1.0,
        lrate_mean: float = 1.0,
        temperature: float = 12.5,
        weight_type: Literal["recomb", "temp"] = "temp",
        device: torch.device | None = None,
    ):
        """Initialize the SNES algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param center_init: The initial center of the population. Must be a 1D tensor.
        :param optimizer: The optimizer to use. Defaults to None. Currently, only "adam" or None is supported.
        :param lrate_mean: The learning rate for the mean. Defaults to 1.0.
        :param sigma: The standard deviation of the noise. Defaults to 1.0.
        :param temperature: The temperature of the softmax in computing weights. Defaults to 12.5.
        :param weight_type: The type of weights to use. Defaults to "temp".
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        assert pop_size > 1
        dim = center_init.shape[0]
        # set hyperparameters
        lrate_sigma = (3 + math.log(dim)) / (5 * math.sqrt(dim))
        self.lrate_mean = Parameter(lrate_mean, device=device)
        self.lrate_sigma = Parameter(lrate_sigma, device=device)
        self.temperature = Parameter(temperature, device=device)
        # set value
        self.dim = dim
        self.pop_size = pop_size
        # setup
        center_init = center_init.to(device=device)

        if weight_type == "temp":
            weights = torch.arange(pop_size, device=device) / (pop_size - 1) - 0.5
            weights = F.softmax(-20 * F.sigmoid(temperature * weights), dim=0)
        if weight_type == "recomb":
            weights = torch.clip(math.log(pop_size / 2 + 1) - torch.log(torch.arange(1, pop_size + 1, device=device)), 0)
            weights = weights / torch.sum(weights) - 1 / pop_size

        weights = torch.tile(weights[:, None], (1, self.dim))

        self.weights = Mutable(weights, device=device)
        self.center = Mutable(center_init)
        self.sigma = Mutable(sigma * torch.ones(self.dim, device=device))

    def step(self):
        """Run one step of the SNES algorithm.

        The function will sample a population, evaluate their fitness, and then
        update the center and standard deviation of the algorithm using the
        sampled population.
        """
        device = self.center.device

        noise = torch.randn(self.pop_size, self.dim, device=device)
        population = self.center + noise * self.sigma.reshape(1, self.dim)

        fitness = self.evaluate(population)

        order = fitness.argsort()
        sorted_noise = noise[order]
        grad_mean = (self.weights * sorted_noise).sum(dim=0)
        grad_sigma = (self.weights * (sorted_noise**2 - 1)).sum(dim=0)

        center = self.center + self.lrate_mean * self.sigma * grad_mean
        sigma = self.sigma * torch.exp(self.lrate_sigma / 2 * grad_sigma)

        self.center = center
        self.sigma = sigma

    def record_step(self):
        return {
            "center": self.center,
            "sigma": self.sigma,
        }
