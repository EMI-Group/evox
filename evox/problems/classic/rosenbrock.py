import jax
import jax.numpy as jnp

import evox as ex


def _rosenbrock_func(x):
    f = jnp.sum(
        100 * ((x[1:])- x[:x.shape[0] - 1] ** 2) ** 2 + (x[:x.shape[0] - 1] - 1) ** 2)
    return f


@ex.jit_class
class Rosenbrock(ex.Problem):
        
    def evaluate(self, state, X):
        return state, jax.vmap(_rosenbrock_func)(X)

