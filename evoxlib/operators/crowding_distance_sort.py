import jax
import jax.numpy as jnp
import chex


@jax.jit
def crowding_distance(x: chex.Array):
    """sort according to crowding distance

    The input x should have rank 2 and should not contain any dominate relation.

    Parameters
    ----------
    x
        An 2d-array.

    Returns
    -------
    ndarray
        An 1d-array containing the crowding distance for x.
    """
    d = jnp.zeros((x.shape[0],))

    def f(carry, fitness):
        rank = jnp.argsort(fitness)
        carry = carry.at[rank[0]].set(jnp.inf)
        carry = carry.at[rank[-1]].set(jnp.inf)
        distance_range = fitness[rank[-1]] - fitness[rank[0]]
        distance = (fitness[rank[2:]] - fitness[rank[:-2]]) / distance_range
        carry = carry.at[rank[1:-1]].add(distance)
        return carry, None

    d, _ = jax.lax.scan(f, d, x.T)
    return d


def crowding_distance_sort(x: chex.Array):
    """sort according to crowding distance

    The input x should have rank 2 and should not contain any dominate relation.

    Parameters
    ----------
    x
        Array to be sorted.

    Returns
    -------
    ndarray
        Array of indices that sort x.
    """
    chex.assert_rank(x, 2)
    distance = crowding_distance(x)
    return jnp.argsort(distance)
