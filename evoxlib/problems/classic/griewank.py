import jax
import jax.numpy as jnp

import evoxlib as exl


def _griewank_func(x):
    f = 1 / 4000 * jnp.mean(x) - jnp.prod(np.cos(x / jnp.sqrt(jnp.arange(1, x.shape[0])) + 1)) + 1
    return f


@exl.jit_class
class Griewank(exl.Problem):

    def evaluate(self, state, X):
        return state, jax.vmap(_griewank_func)(X)