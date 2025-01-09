from typing import Literal

import torch
import torch.nn.functional as F

from ...core import Parameter, Mutable, Algorithm, jit_class
from .adam_step import adam_single_tensor

@jit_class
class PersistentES(Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: torch.Tensor,
        optimizer: Literal[ "adam" ] | None = None,
        lr: float = 0.05,
        sigma: float = 0.03,
        T: int = 100,
        K: int = 10,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        device: torch.device | None = None,
    ):
        super().__init__()
        
        assert pop_size % 2 == 0  # Population size must be even

        dim = center_init.shape[0]

        # set hyperparameters
        self.lr = Parameter( lr, device=device )
        self.T = Parameter( T, device=device ) 
        self.K = Parameter( K, device=device )
        self.sigma_decay = Parameter( sigma_decay, device=device )
        self.sigma_limit = Parameter( sigma_limit, device=device )
        # set value
        self.dim = dim
        self.pop_size = pop_size
        self.optimizer = optimizer
        # setup
        center_init = center_init.to( device=device )
        self.sigma = Mutable( torch.tensor(sigma) )
        self.center = Mutable(center_init)
        self.inner_step_counter = Mutable( torch.tensor(0.0) )
        self.pert_accum = Mutable( torch.zeros( pop_size, dim, device=device) )

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
        pert_accum = self.pert_accum + perts
        population = self.center + perts
        
        fitness = self.evaluate( population )

        theta_grad = torch.mean( pert_accum * fitness.reshape(-1, 1) / (self.sigma**2), dim=0 )
        
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
        self.inner_step_counter=inner_step_counter
        
        reset = self.inner_step_counter >= self.T
        inner_step_counter = torch.where( reset, 0, inner_step_counter)
        pert_accum = torch.where( reset, torch.zeros( self.pop_size, self.dim, device=device ), pert_accum )

        sigma = self.sigma_decay * self.sigma
        sigma = torch.maximum(sigma, self.sigma_limit)

        self.sigma=sigma
        self.pert_accum=pert_accum
