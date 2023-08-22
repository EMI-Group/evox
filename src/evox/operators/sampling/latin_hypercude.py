import jax
import jax.numpy as jnp
from evox import jit_class


@jit_class
class LatinHypercubeSampling:
    """Latin hypercube sampling"""

    def __init__(self, n=None, m=None):
        self.n = n
        self.m = m

    def __call__(self, key):
        subkeys = jax.random.split(key, self.m)
        w = jax.random.uniform(key, shape=(self.n, self.m))
        parm = jnp.tile(jnp.arange(1, self.n + 1), (self.m, 1))
        parm = jax.vmap(jax.random.permutation, in_axes=(0, 0), out_axes=1)(
            subkeys, parm
        )
        w = (parm - w) / self.n
        n = self.n
        return w, n
