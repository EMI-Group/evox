import jax
import jax.numpy as jnp
from evox import Problem, jit_class


def _griewank_func(x):
    f = 1 / 4000 * jnp.mean(x) - jnp.prod(jnp.cos(x / jnp.sqrt(jnp.arange(1, x.shape[0] + 1)) )) + 1
    return f


@jit_class
class Griewank(Problem):

    def evaluate(self, state, X):
        return jax.vmap(_griewank_func)(X), state