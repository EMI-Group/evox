import chex
from typing import Union
import jax.numpy as jnp


def min_by(
    values: Union[chex.Array | list[chex.Array]],
    keys: Union[chex.Array | list[chex.Array]],
):
    if isinstance(values, list):
        values = jnp.concatenate(values)
        keys = jnp.concatenate(keys)

    min_index = jnp.argmin(keys)
    return values[min_index], keys[min_index]
