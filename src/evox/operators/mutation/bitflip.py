import jax
import jax.numpy as jnp
from evox import jit_class, Operator, State


def _flip(key, x, p):
    probabilities = jax.random.uniform(key, x.shape)
    new_x = jnp.where(probabilities < p, x, ~x)
    return new_x

@jit_class
class BitFlipMutation(Operator):
    def __init__(self, p):
        """
        Parameters
        ----------
        p
            The probability to flip each bit.
        """
        self.p = p

    def setup(self, key):
        return State(key=key)

    def __call__(self, state, x):
        batch, _ = x.shape
        key, subkey = jax.random.split(state.key)
        mutation_keys = jax.random.split(subkey, batch)
        ps = jnp.ones(shape=(batch,)) * self.p
        new_x = jax.vmap(_flip)(mutation_keys, x, ps)
        return new_x, State(key=key)
