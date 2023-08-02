import jax
import jax.numpy as jnp
from evox import jit_class, Operator, State


@jit_class
class RouletteWheelSelection(Operator):
    """Roulette wheel selection"""

    def __init__(self, n):
        self.n = n

    def setup(self, key):
        return State(key=key)

    def __call__(self, state, fitness):
        key, subkey = jax.random.split(state.key)

        fitness = fitness - jnp.minimum(jnp.min(fitness), 0) + 1e-6
        fitness = jnp.cumsum(1. / fitness)
        fitness = fitness / jnp.max(fitness)

        index = jnp.zeros(self.n).astype(int)
        vals = (index, key)

        def body_fun(i, vals):
            index, key = vals
            key, subkey = jax.random.split(key)
            r = jax.random.uniform(subkey)
            idx = jnp.argwhere(fitness >= r, size=10)
            index = index.at[i].set(idx[0, 0])
            val = (index, key)
            return val

        index, key = jax.lax.fori_loop(0, self.n, body_fun, vals)

        return index, State(key=key)
