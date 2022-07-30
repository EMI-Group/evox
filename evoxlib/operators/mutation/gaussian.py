import evoxlib as exl
import jax
import jax.numpy as jnp


class GaussianMutation(exl.Operator):
    def __init__(self, stdvar=1.0):
        self.stdvar = stdvar

    def setup(self, key):
        return {
            'key': key
        }

    def __call__(self, state, x):
        key, subkey = jax.random.split(state['key'])
        perturbation = jax.random.normal(subkey, x.shape) * self.stdvar
        return {"key": key}, x + perturbation