<<<<<<< HEAD
import jax
from evox import jit_class, Operator, State


@jit_class
class GaussianMutation(Operator):
    def __init__(self, stdvar=1.0):
        self.stdvar = stdvar

    def setup(self, key):
        return State(key=key)

    def __call__(self, state, x):
        key, subkey = jax.random.split(state.key)
        perturbation = jax.random.normal(subkey, x.shape) * self.stdvar
        return x + perturbation, State(key=key)
=======
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
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
