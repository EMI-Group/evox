import jax
import jax.numpy as jnp

import evoxlib as exl


def _rosenbrock_func(x):
    f = jnp.sum(
        100 * ((x[1:])- x[:x.shape[0] - 1] ** 2) ** 2 + (x[:x.shape[0] - 1] - 1) ** 2)
    return f


@exl.jit_class
class Rosenbrock(exl.Problem):
        
    def evaluate(self, state, X):
        return state, jax.vmap(_rosenbrock_func)(X)

