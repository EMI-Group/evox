import jax
import jax.numpy as jnp
import chex


@jax.jit
def _dominate(x, y):
    """return true if x dominate y (x < y) and false elsewise.
    """
    return jnp.all(x <= y) & jnp.any(x < y)


@jax.jit
def _dominate_relation(x, y):
    """return a matrix A, where A_{ij} is True if x_i donminate y_j
    """
    return jax.vmap(lambda _x: jax.vmap(lambda _y: _dominate(_x, _y))(y))(x)


def non_dominated_sort(x):
    """ Perform non-dominated sort

    Currently JAX doesn't support dynamic shape with jit,
    so part of the function must be runned without jit
    """
    chex.assert_rank(x, 2)
    dominate_relation_matrix = _dominate_relation(x, x)
    rank = jnp.zeros((x.shape[0],), dtype=jnp.int32)
    dominate_count = jnp.sum(dominate_relation_matrix, axis=0)

    with jax.disable_jit():
        current_rank = 0
        next_front = dominate_count == 0
        while jnp.any(next_front):
            rank = rank.at[next_front].set(current_rank)
            _, index_to_decrease = jnp.where(dominate_relation_matrix[next_front, :] == True)
            dominate_count = dominate_count.at[
                index_to_decrease
            ].add(-1)
            dominate_count = dominate_count.at[next_front].add(-1)

            current_rank+=1
            next_front = dominate_count == 0
    return rank
