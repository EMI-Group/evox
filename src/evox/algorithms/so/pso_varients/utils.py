import jax
import jax.numpy as jnp
from jax import vmap

from typing import Union, List


def min_by(
    values: Union[jax.Array, List[jax.Array]],
    keys: Union[jax.Array, List[jax.Array]],
):
    if isinstance(values, list):
        values = jnp.concatenate(values)
        keys = jnp.concatenate(keys)

    min_index = jnp.argmin(keys)
    return values[min_index], keys[min_index]


def get_distance_matrix(
    location: jax.Array,
):
    """
    N*M matrix indicating N locations over M dimensions
    return a N*N distance matrix
    """
    assert len(location.shape) == 2

    def dist(x, y):
        return jax.numpy.linalg.norm(x - y)

    return jax.vmap(lambda x: jax.vmap(lambda y: dist(x, y))(location))(location)


def row_argsort(
    x: jax.Array,
):
    assert len(x.shape) == 2
    """
    x is a N*M matrix
    return a N*M matrix of indices that sort each row of x
    """
    y = jnp.argsort(
        x, axis=-1, kind="stable"
    )  # sort each row of x and get a N*M index array
    return y


@jax.jit
def select_from_mask(key: jax.Array, mask: jax.Array, s: int):
    """
    given a 1d array with {0,1} mask, randomly choose s element from mask

    # thanks wang li shuang for help
    """
    N = mask.shape[0]
    noise = jax.random.uniform(key, (N,))
    sorted_idx = jnp.argsort(mask + noise)[::-1]
    sorted_idx = jnp.where(jnp.arange(N) < s, sorted_idx, sorted_idx[s - 1])
    return jnp.zeros_like(mask).at[sorted_idx].set(1)
