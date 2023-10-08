import jax
import jax.numpy as jnp
from evox import Problem, jit_class


def _rosenbrock_func(x):
    f = jnp.sum(
        100 * ((x[1:])- x[:x.shape[0] - 1] ** 2) ** 2 + (x[:x.shape[0] - 1] - 1) ** 2)
    return f

def rosenbrock_func(X):
    return jax.vmap(_rosenbrock_func)(X)


@jit_class
class Rosenbrock(Problem):
    def evaluate(self, state, X):
        return rosenbrock_func(X), state
