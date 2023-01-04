import evox as ex
import jax
import jax.numpy as jnp

def _flip(key, x, p):
    probabilities = jax.random.uniform(key, x.shape)
    new_x = jnp.where(probabilities < p, x, ~x)
    return new_x

@ex.jit_class
class BitFlipMutation(ex.Operator):
    def __init__(self, p):
        """
        Parameters
        ----------
        p
            The probability to flip each bit.
        """
        self.p = p

    def setup(self, key):
        return ex.State(key=key)

    def __call__(self, state, x):
        batch, _ = x.shape
        key, subkey = jax.random.split(state.key)
        mutation_keys = jax.random.split(subkey, batch)
        ps = jnp.ones(shape=(batch,)) * self.p
        new_x = jax.vmap(_flip)(mutation_keys, x, ps)
        return ex.State(key=key), new_x
