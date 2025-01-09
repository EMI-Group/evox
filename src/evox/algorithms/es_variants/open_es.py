from typing import Literal

import torch

from ...core import Parameter, Mutable, Algorithm, jit_class
from .adam_step import adam_single_tensor


@jit_class
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
        """
        Initialize the PSO algorithm with the given parameters.

        Args:
            pop_size (`int`): The size of the population.
            center_init (`torch.Tensor` or `Dict[str, torch.Tensor]`): The initial center of the population. Must be a 1D tensor.
            learning_rate (`float`): The learning rate for the optimizer.
            noise_stdev (`float`): The standard deviation of the noise.
            optimizer (`str`, optional): The optimizer to use. Defaults to None. Currently, only "adam" or None is supported.
            mirrored_sampling (`bool`, optional): Whether to use mirrored sampling. Defaults to True.
            device (`torch.device`, optional): The device to use for the tensors. Defaults to None.
        """
        super().__init__()
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
        device = self.center.device
        if self.mirrored_sampling:
            noise = torch.randn(self.pop_size // 2, self.dim, device=device)
            noise = torch.cat([noise, -noise], dim=0)
        else:
            noise = torch.randn(self.pop_size, self.dim, device=device)
        population = self.center[None, :] + self.noise_stdev * noise
        fitness = self.evaluate(population)
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
