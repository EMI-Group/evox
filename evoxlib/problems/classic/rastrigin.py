import jax
import jax.numpy as jnp

import evoxlib as exl


def _rastrigin_func(x):
    return 10 * x.shape[0] + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))


@exl.jit_class
class Rastrigin(exl.Problem):
    def evaluate(self, state, X):
        return state, jax.vmap(_rastrigin_func)(X)
