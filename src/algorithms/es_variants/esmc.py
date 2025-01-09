from typing import Literal

import torch

from ...core import Parameter, Mutable, Algorithm, jit_class
from .adam_step import adam_single_tensor


@jit_class
class ESMC(Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: torch.Tensor,
        optimizer: Literal[ "adam" ] | None = None,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        lr: float = 0.05,
        sigma: float = 0.03,
        device: torch.device | None = None,
    ):
        super().__init__()
        
        assert pop_size & 1
        
        dim = center_init.shape[0]
        
        # set hyperparameters
        self.lr = Parameter(lr,device=device)
        self.sigma_decay = Parameter( sigma_decay, device=device )
        self.sigma_limit = Parameter( sigma_limit, device=device )
        # set value
        self.dim = dim
        self.pop_size = pop_size
        self.optimizer = optimizer
        # setup
        center_init = center_init.to( device=device )
        self.center = Mutable( center_init )
        self.sigma = Mutable( torch.ones(self.dim) * sigma )

        if optimizer == "adam":
            self.exp_avg = Mutable(torch.zeros_like(self.center))
            self.exp_avg_sq = Mutable(torch.zeros_like(self.center))
            self.beta1 = Parameter(0.9, device=device)
            self.beta2 = Parameter(0.999, device=device)

    def step(self):
        device = self.center.device
        
        z_plus = torch.randn( int(self.pop_size / 2), self.dim, device=device )
        z = torch.cat( [torch.zeros( 1, self.dim, device=device), z_plus, -1.0 * z_plus])
        
        population = self.center + z * self.sigma.reshape(1, self.dim)
        
        fitness = self.evaluate( population )

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
