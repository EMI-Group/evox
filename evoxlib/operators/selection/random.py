import evoxlib as exl

import jax
import jax.numpy as jnp

class RandomSelection(exl.Operator):
    def __init__(self, p):
        self.p = p

    def setup(self, key):
        return {
            "key": key
        }

    def __call__(self, state, x):
        key, subkey = jax.random.split(state["key"])
        num = int(x.shape[0] * self.p)
        chosen = jax.random.choice(subkey, x.shape[0], shape=(num, ))
        return {"key": key}, x[chosen, :]