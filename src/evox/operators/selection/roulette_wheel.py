import jax
import jax.numpy as jnp
from evox import jit_class


@jit_class
class RouletteWheelSelection:
    """Roulette wheel selection

    The smaller the fitness, the greater the selection probability.
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, key, x, fitness):

        fitness = fitness - jnp.minimum(jnp.min(fitness), 0) + 1e-6
        fitness = jnp.cumsum(1.0 / fitness)
        fitness = fitness / jnp.max(fitness)

        random_values = jax.random.uniform(key, shape=(self.n,))

        selected_indices = jnp.searchsorted(fitness, random_values)

        return x[selected_indices], selected_indices
