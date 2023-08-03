<<<<<<< HEAD
import jax
import jax.numpy as jnp
from evox import jit_class, Operator, State

=======
from evox import jit_class
import jax.numpy as jnp
from jax import jit, random
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9


<<<<<<< HEAD
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
=======
@jit
def bitflip(key, x, prob, bool_input="auto"):
    """Perform bitflip mutation, the input x is expected to have a dtype of bool or uint8.
    The mutation can be performed in either bool mode or packed bits mode.
    If bool_input is True, the input is interpreted as a boolean tensor,
    If bool_input is False, the input is interpreted as a uint8 tensor where each number represents 8 bits.
    If bool_input is "auto", it is inferred from x.dtype.
    """
    if bool_input == "auto":
        if x.dtype == jnp.bool_:
            bool_input = True
        elif x.dtype == jnp.uint8:
            bool_input = False
        else:
            raise "The input x should be type bool or uint8"
    # xor True will flip the bit, and xor False won't
    if bool_input:
        flips = random.uniform(key, x.shape) < prob
        return x ^ flips
    elif x.dtype == jnp.uint8:
        flips = random.uniform(key, (*x.shape, 8)) < prob
        flips = jnp.packbits(flips, axis=-1)
        return x ^ flips


@jit_class
class Bitflip:
    def __init__(self, prob, bool_input="auto"):
        self.prob = prob
        self.bool_input = bool_input

    def __call__(self, key, x):
        return bitflip(key, x, self.prob, self.bool_input)
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
