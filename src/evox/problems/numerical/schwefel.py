import jax
from jax import jit
import jax.numpy as jnp
from evox import Problem, jit_class


@jit
def schwefel_func(x):
    _pop_size, dim = x.shape
    return 418.9828872724338 * dim - jnp.sum(x * jnp.sin(jnp.sqrt(jnp.abs(x))), axis=1)


@jit_class
class Schwefel(Problem):
    """The Schwefel function
    The minimum is x = [420.9687462275036, ...]
    """

    def __init__(self):
        super().__init__()

    def evaluate(self, state, x):
        return schwefel_func(x), state
