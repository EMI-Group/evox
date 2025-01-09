from typing import Literal

import math
import torch
import torch.nn.functional as F

from ...core import Parameter, Mutable, Algorithm, jit_class
from ...utils import clamp



@jit_class
class SNES(Algorithm):
    def __init__(
        self,
        pop_size: int,
        center_init: torch.Tensor,
        sigma: float = 1.0,
        lrate_mean: float = 1.0,
        temperature: float = 12.5,
        weight_type: Literal[ "recomb", "temp" ] = "temp",
        device: torch.device | None = None,
    ):
        super().__init__()
        
        dim = center_init.shape[0]

        # set hyperparameters
        lrate_sigma = (3 + math.log(dim)) / (5 * math.sqrt(dim))
        self.lrate_mean = Parameter( lrate_mean, device=device )
        self.lrate_sigma = Parameter( lrate_sigma, device=device )
        self.temperature = Parameter( temperature, device=device )
        # set value
        self.dim = dim
        self.pop_size = pop_size
        # setup
        center_init = center_init.to( device=device )
        
        if weight_type == "temp" :
            weights = torch.arange( pop_size, device=device) / ( pop_size - 1 ) - 0.5
            weights = F.softmax( - 20 * F.sigmoid( temperature * weights ), dim = 0 )
        if weight_type == "recomb" :
            weights = torch.clip( math.log(pop_size / 2 + 1) - torch.log( torch.arange( 1, pop_size + 1, device=device ) ), 0 )
            weights = weights / torch.sum(weights) - 1 / pop_size
        
        weights = torch.tile( weights[ :, torch.newaxis ], ( 1, self.dim ) )
        
        self.weights = Mutable(weights)
        self.center = Mutable(center_init)
        self.sigma = Mutable(sigma * torch.ones( self.dim, device=device ))

    def step(self):
        device = self.center.device
        
        noise = torch.randn( self.pop_size, self.dim, device=device )
        population = self.center + noise * self.sigma.reshape( 1, self.dim )
        
        fitness = self.evaluate( population )
        
        order = fitness.argsort()
        sorted_noise = noise[order]
        grad_mean = ( self.weights * sorted_noise ).sum( dim = 0 )
        grad_sigma = ( self.weights * (sorted_noise ** 2 - 1) ).sum( dim = 0 )
        
        center = self.center + self.lrate_mean * self.sigma * grad_mean
        sigma = self.sigma * torch.exp(self.lrate_sigma / 2 * grad_sigma)
        
        self.center = center
        self.sigma = sigma