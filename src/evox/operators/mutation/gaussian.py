import evox as ex
import jax
import jax.numpy as jnp


@ex.jit_class
class GaussianMutation(ex.Operator):
    def __init__(self, stdvar=1.0):
        self.stdvar = stdvar

    def setup(self, key):
        return ex.State(key=key)

    def __call__(self, state, x):
        key, subkey = jax.random.split(state.key)
        perturbation = jax.random.normal(subkey, x.shape) * self.stdvar
        return ex.State(key=key), x + perturbation
