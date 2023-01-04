import evox as ex

import jax
import jax.numpy as jnp


@ex.jit_class
class UniformRandomSelection(ex.Operator):
    def __init__(self, p):
        self.p = p

    def setup(self, key):
        return ex.State(key=key)

    def __call__(self, state, x):
        key, subkey = jax.random.split(state.key)
        num = int(x.shape[0] * self.p)
        chosen = jax.random.choice(subkey, x.shape[0], shape=(num,))
        return ex.State(key=key), x[chosen, :]
