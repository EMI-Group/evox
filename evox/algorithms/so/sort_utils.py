import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def sort_key_valrows(keys, *vals):
    assert(len(keys.shape) == 1)
    keys, perm = lax.sort((keys, jnp.arange(0, keys.shape[0])), is_stable=False)
    vals = map(lambda v: v[perm], vals)
    return keys, *vals