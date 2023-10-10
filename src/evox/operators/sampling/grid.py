import jax
import jax.numpy as jnp
from itertools import combinations as n_choose_k
from scipy.special import comb

import evox


@evox.jit_class
class GridSampling:
    """
    Grid sampling.
    Inspired by PlatEMO's NBI algorithm.
    """

    def __init__(self, n=None, m=None):
        self.n = n
        self.m = m
        self.num_points = int(jnp.ceil(self.n ** (1 / self.m)).item())

    def __call__(self):
        gap = jnp.linspace(0, 1, self.num_points)
        grid_axes = [gap for _ in range(self.m)]
        grid_values = jnp.meshgrid(*grid_axes, indexing='ij') # Equivalent of MATLAB's ndgrid
        w = jnp.stack(grid_values, axis=-1).reshape(-1, self.m)
        w = w[:, ::-1]
        n = w.shape[0]
        return w, n
