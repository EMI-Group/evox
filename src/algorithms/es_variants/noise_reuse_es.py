from typing import Literal

import torch

from ...core import Parameter, Mutable, Algorithm, jit_class
from .adam_step import adam_single_tensor


@jit_class
class Noise_reuse_es(Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: torch.Tensor,
        optimizer: Literal[ "adam" ] | None = None,
        lr: float = 0.05,
        sigma: float = 0.03,
        T: int = 100,  # inner problem length
        K: int = 10,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        device: torch.device | None = None,
    ):
        super().__init__()

        dim = center_init.shape[0]

        
        # set hyperparameters
        self.lr = Parameter(lr,device=device)
        self.T = Parameter(T,device=device)
        self.K = Parameter(K,device=device)
        self.sigma_decay = Parameter(sigma_decay,device=device)
        self.sigma_limit = Parameter(sigma_limit,device=device)
        # set value
        self.dim = dim
        self.pop_size = pop_size
        self.optimizer = optimizer
        # setup
        center_init = center_init.to( device=device )
        self.center = Mutable(center_init)
        self.sigma = Mutable( torch.tensor(sigma) )
        self.inner_step_counter = Mutable( torch.tensor(0.0) )
        self.unroll_pert = Mutable( torch.zeros(pop_size, self.dim, device=device) )
        
        if optimizer == "adam":
            self.exp_avg = Mutable(torch.zeros_like(self.center))
            self.exp_avg_sq = Mutable(torch.zeros_like(self.center))
            self.beta1 = Parameter(0.9, device=device)
            self.beta2 = Parameter(0.999, device=device)

    def step(self):
        device = self.center.device
        
        pos_perts = torch.randn( self.pop_size // 2, self.dim, device=device ) * self.sigma
        neg_perts = -pos_perts
        perts = torch.cat([pos_perts, neg_perts], dim=0)
        unroll_pert = torch.where( self.inner_step_counter == 0, perts, self.unroll_pert)
        
        population = self.center + unroll_pert
        
        fitness = self.evaluate( population )

        theta_grad = torch.mean( unroll_pert * fitness.reshape(-1, 1) / (self.sigma**2), dim=0 )
        
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
        
        inner_step_counter = torch.where( self.inner_step_counter + self.K >= self.T, 0, self.inner_step_counter + self.K )
        self.inner_step_counter = inner_step_counter
        
        sigma = torch.maximum( self.sigma_decay * self.sigma, self.sigma_limit )
        self.sigma = sigma
