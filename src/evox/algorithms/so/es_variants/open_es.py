from typing import Literal

import torch

from evox.core import Algorithm, Mutable, Parameter

from .adam_step import adam_single_tensor


class OpenES(Algorithm):
    """The OpenES algorithm as described in "Evolution Strategies as a Scalable Alternative to Reinforcement Learning" from https://arxiv.org/abs/1703.03864."""

    def __init__(
        self,
        pop_size: int,
        # TODO: we will support a parameter dictionary as center_init in the future
        center_init: torch.Tensor,
        learning_rate: float,
        noise_stdev: float,
        optimizer: Literal["adam"] | None = None,
        mirrored_sampling: bool = True,
        device: torch.device | None = None,
    ):
        """Initialize the OpenES algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param center_init: The initial center of the population. Must be a 1D tensor.
        :param learning_rate: The learning rate for the optimizer.
        :param noise_stdev: The standard deviation of the noise.
        :param optimizer: The optimizer to use. Defaults to None. Currently, only "adam" or None is supported.
        :param mirrored_sampling: Whether to use mirrored sampling. Defaults to True.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        assert noise_stdev > 0, "noise_stdev must be greater than 0."
        assert learning_rate > 0, "learning_rate must be greater than 0."
        assert pop_size > 0, "pop_size must be greater than 0."
        if mirrored_sampling is True:
            assert pop_size % 2 == 0, "When mirrored_sampling is True, pop_size must be a multiple of 2."
        assert optimizer in [None, "adam"], "optimizer must be None or 'adam'."
        # set hyperparameters
        self.learning_rate = Parameter(learning_rate, device=device)
        self.noise_stdev = Parameter(noise_stdev, device=device)
        # set value
        center_init = center_init.to(device=device)
        self.pop_size = pop_size
        self.dim = center_init.shape[0]
        self.mirrored_sampling = mirrored_sampling
        self.optimizer = optimizer
        # setup
        self.center = Mutable(center_init)
        # TODO: use submodule instead of string for optimizer in the future
        if optimizer == "adam":
            self.exp_avg = Mutable(torch.zeros_like(self.center))
            self.exp_avg_sq = Mutable(torch.zeros_like(self.center))
            self.beta1 = Parameter(0.9, device=device)
            self.beta2 = Parameter(0.999, device=device)

    def step(self):
        """Step the OpenES algorithm by evaluating the fitness of the current population and updating the center."""
        device = self.center.device
        if self.mirrored_sampling:
            noise = torch.randn(self.pop_size // 2, self.dim, device=device)
            noise = torch.cat([noise, -noise], dim=0)
        else:
            noise = torch.randn(self.pop_size, self.dim, device=device)
        pop = self.center[None, :] + self.noise_stdev * noise
        fitness = self.evaluate(pop)
        grad = noise.T @ fitness / self.pop_size / self.noise_stdev
        if self.optimizer is None:
            center = self.center - self.learning_rate * grad
        else:
            center, self.exp_avg, self.exp_avg_sq = adam_single_tensor(
                self.center,
                grad,
                self.exp_avg,
                self.exp_avg_sq,
                self.beta1,
                self.beta2,
                self.learning_rate,
            )
        self.center = center

    def record_step(self):
        return {"center": self.center}
