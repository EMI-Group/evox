from functools import partial

import jax
import jax.numpy as jnp
import copy

import evoxlib as exl


@exl.jit_class
class PGPE(exl.Algorithm):
    def __init__(self, pop_size, center_init, stdev_init, stdev_learning_rate):
        self.dim = center_init.shape[0]
        self.pop_size = pop_size
        self.stdev = stdev_init
        self.stdev_learning_rate = stdev_learning_rate

    def setup(self, key):
        return exl.State(
            center = self.center_init,
            stdev = self.stdev,
            key = key
        )

    def ask(self, state):
        key, subkey = jax.random.split(state.key)
        delta = jax.random.normal(state.key, (self.pop_size // 2, self.dim)) * state.stdev
        D = jnp.concatenate([state.center + delta, state.center - delta])
        return state.update(key = subkey), D

    def tell(self, state, x, F):
        D_pos = x[:self.pop_size // 2, :]
        D_neg = x[self.pop_size // 2:, :]
        delta_x = jnp.mean((D_pos - D_neg) / 2 * (D_pos - state.center))
