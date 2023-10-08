import jax
import jax.numpy as jnp
from evox import Problem, jit_class


def _rastrigin_func(x):
    return 10 * x.shape[0] + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))

def rastrigin_func(X):
    return jax.vmap(_rastrigin_func)(X)

@jit_class
class Rastrigin(Problem):
    def evaluate(self, state, X):
        return rastrigin_func(X), state
