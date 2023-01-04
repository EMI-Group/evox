from functools import partial
import jax
import jax.numpy as jnp

import evox as ex


def _ackley_func(a, b, c, x):
    return (
        -a * jnp.exp(-b * jnp.sqrt(jnp.mean(x**2)))
        - jnp.exp(jnp.mean(jnp.cos(c * x)))
        + a
        + jnp.e
    )


@ex.jit_class
class Ackley(ex.Problem):
    def __init__(self, a=20, b=0.2, c=2):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, state, X):
        return state, jax.vmap(partial(_ackley_func, self.a, self.b, self.c))(X)
