import jax
import jax.numpy as jnp
from evox import jit_class, Operator, State


@jit_class
class UniformRandomSelection(Operator):
    def __init__(self, p):
        self.p = p

    def setup(self, key):
        return State(key=key)

    def __call__(self, state, x):
        key, subkey = jax.random.split(state.key)
        num = int(x.shape[0] * self.p)
        chosen = jax.random.choice(subkey, x.shape[0], shape=(num,))
        return chosen, State(key=key)
