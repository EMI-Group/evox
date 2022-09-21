import jax
import jax.numpy as jnp

import evoxlib as exl


def _rosenbrock_func(x):
    f = jnp.sum(
        100 * (x[:, :x.shape[1] - 1]) ** 2 - x[:, 1:x.shape[1]] ** 2 + (x[:, :x.shape[1] - 1] - 1) ** 2,
        axis=1, keepdims=True)
    return f


@exl.jit_class
class Rosenbrock(exl.Problem):
        
    def evaluate(self, state, X):
        return state, jax.vmap(_rosenbrock_func)(X)