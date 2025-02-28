from typing import Literal

import torch

from evox.core import Algorithm, Mutable, Parameter

from .adam_step import adam_single_tensor


class NoiseReuseES(Algorithm):
    """The implementation of the Noise-Reuse-ES algorithm.

    Reference:
    Noise-Reuse in Online Evolution Strategies
    (https://arxiv.org/pdf/2304.12180.pdf)

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
        T: int = 100,  # inner problem length
        K: int = 10,
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
        :param T: The inner problem length. Defaults to 100.
        :param K: The number of inner problems. Defaults to 10.
        :param device: The device to use for the tensors. Defaults to None.
        """
        super().__init__()
        assert pop_size > 1
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
        self.center = Mutable(center_init)
        self.sigma = Mutable(torch.tensor(sigma))
        self.inner_step_counter = Mutable(torch.tensor(0.0, device=device))
        self.unroll_pert = Mutable(torch.zeros(pop_size, self.dim, device=device))

        if optimizer == "adam":
            self.exp_avg = Mutable(torch.zeros_like(self.center))
            self.exp_avg_sq = Mutable(torch.zeros_like(self.center))
            self.beta1 = Parameter(0.9, device=device)
            self.beta2 = Parameter(0.999, device=device)

    def step(self):
        """
        Take a single step of the NoiseReuseES algorithm.

        This function follows the algorithm described in the reference paper.
        It first generates a set of perturbations for the current population.
        Then, it evaluates the fitness of the population with the perturbations.
        Afterwards, it calculates the gradient of the policy parameters using the
        perturbations and the fitness.
        Finally, it updates the policy parameters using the gradient and the
        learning rate.
        """
        device = self.center.device

        position_perturbations = torch.randn(self.pop_size // 2, self.dim, device=device) * self.sigma
        negative_perturbations = -position_perturbations
        perturbations = torch.cat([position_perturbations, negative_perturbations], dim=0)
        unroll_pert = torch.where(self.inner_step_counter == 0, perturbations, self.unroll_pert)

        population = self.center + unroll_pert

        fitness = self.evaluate(population)

        theta_grad = torch.mean(unroll_pert * fitness.reshape(-1, 1) / (self.sigma**2), dim=0)

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

        inner_step_counter = torch.where(self.inner_step_counter + self.K >= self.T, 0, self.inner_step_counter + self.K)
        self.inner_step_counter = inner_step_counter

        sigma = torch.maximum(self.sigma_decay * self.sigma, self.sigma_limit)
        self.sigma = sigma

    def record_step(self):
        return {"center": self.center, "sigma": self.sigma}
