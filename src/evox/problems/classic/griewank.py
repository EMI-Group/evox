import jax
import jax.numpy as jnp

import evox as ex


def _griewank_func(x):
    f = 1 / 4000 * jnp.mean(x) - jnp.prod(jnp.cos(x / jnp.sqrt(jnp.arange(1, x.shape[0] + 1)) )) + 1
    return f


@ex.jit_class
class Griewank(ex.Problem):

    def evaluate(self, state, X):
        return state, jax.vmap(_griewank_func)(X)