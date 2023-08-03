from typing import Any
import jax
from jax import jit
from evox import jit_class


@jit
def gaussian(key, x, stdvar):
    perturbation = jax.random.normal(key, x.shape) * stdvar
    return x + perturbation


class Gaussian:
    def __init__(self, stdvar=1.0):
        self.stdvar = stdvar

    def __call__(self, key, x):
        return gaussian(key, x, self.stdvar)
