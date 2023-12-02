import jax
import jax.numpy as jnp


@jax.jit
def crowding_distance(costs: jax.Array, mask: jax.Array = None):
    """sort according to crowding distance

    The input x should have shape (n, d), and mask is None or
    an boolean array with shape(n, ) where True means the corresponding
    element should participate in the whole process, and False means that
    element is ignored.

    Parameters
    ----------
    x
        An 2d-array.
    mask
        An 1d-boolean-array

    Returns
    -------
    ndarray
        An 1d-array containing the crowding distance for x. Ignored elements have distance -inf.
    """
    totel_len = costs.shape[0]
    if mask is None:
        num_valid_elem = totel_len
        mask = jnp.ones(totel_len, dtype=jnp.bool_)
    else:
        num_valid_elem = jnp.sum(mask)

    def distance_in_one_dim(cost):
        rank = jnp.lexsort((cost, ~mask))
        cost = cost[rank]
        distance_range = cost[num_valid_elem-1] - cost[0]
        distance = jnp.empty(totel_len)
        distance = distance.at[rank[1:-1]].set((cost[2:] - cost[:-2]) / distance_range)
        distance = distance.at[rank[0]].set(jnp.inf)
        distance = distance.at[rank[num_valid_elem-1]].set(jnp.inf)

        distance = jnp.where(mask, distance, -jnp.inf)
        return distance

    return jnp.sum(jax.vmap(distance_in_one_dim, 1, 1)(costs), axis=1)


@jax.jit
def crowding_distance_sort(x: jax.Array, mask: jax.Array = None):
    """sort according to crowding distance

    The input x should have rank 2 and should not contain any dominate relation.

    Parameters
    ----------
    x
        Array to be sorted.
    mask
        An 1d-boolean-array

    Returns
    -------
    ndarray
        Array of indices that sort x.
    """
    distance = crowding_distance(x, mask)
    return jnp.argsort(distance)
