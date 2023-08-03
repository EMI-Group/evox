import jax
import jax.numpy as jnp
from evox import jit_class
from jax.experimental.host_callback import id_print


@jit_class
class RouletteWheelSelection:
    """Roulette wheel selection

    The smaller the fitness, the greater the selection probability.
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, key, x, fitness):

        fitness = fitness - jnp.minimum(jnp.min(fitness), 0) + 1e-6
        fitness = jnp.cumsum(1. / fitness)
        fitness = fitness / jnp.max(fitness)
        # id_print(fitness)

        random_values = jax.random.uniform(key, shape=(self.n, ))
        # id_print(random_values)
        selected_indices = jnp.searchsorted(fitness, random_values)

        return x[selected_indices], selected_indices


if __name__ == '__main__':
    r = RouletteWheelSelection(10)
    x = jnp.array([3.0, 1.5, 2.7, 5.2, 4.1])
    idx = r(jax.random.PRNGKey(22), x, x)
    print(idx)
