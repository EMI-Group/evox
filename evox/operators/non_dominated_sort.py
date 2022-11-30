import jax
import jax.numpy as jnp
import chex


@jax.jit
def _dominate(x, y):
    """return true if x dominate y (x < y) and false elsewise."""
    if jnp.issubdtype(x, jnp.inexact) and jnp.issubdtype(y, jnp.inexact):
        return jnp.all(x < y) # don't need to check for equal relation
    else:
        return jnp.all(x <= y) & jnp.any(x < y)


@jax.jit
def _dominate_relation(x, y):
    """return a matrix A, where A_{ij} is True if x_i donminate y_j"""
    return jax.vmap(lambda _x: jax.vmap(lambda _y: _dominate(_x, _y))(y))(x)


@jax.jit
def non_dominated_sort(x):
    """Perform non-dominated sort

    Currently JAX doesn't support dynamic shape with jit,
    so part of the function must be runned without jit
    """
    chex.assert_rank(x, 2)
    dominate_relation_matrix = _dominate_relation(x, x)
    rank = jnp.zeros((x.shape[0],), dtype=jnp.int32)
    dominate_count = jnp.sum(dominate_relation_matrix, axis=0)

    indices = jnp.indices((x.shape[0], ))
    empty = x.shape[0] + 1

    current_rank = 0
    next_front = dominate_count == 0

    def _cond_fun(loop_state):
        _rank, _dominate_count, _current_rank, next_front = loop_state
        return jnp.any(next_front)

    def _body_fun(loop_state):
        rank, dominate_count, current_rank, next_front = loop_state
        rank = jnp.where(next_front, current_rank, rank)
        masked_dominate_relation_matrix = jnp.where(next_front[:, jnp.newaxis], dominate_relation_matrix, 0)
        indices_to_decs = jnp.where(masked_dominate_relation_matrix, indices, empty)
        dominate_count = dominate_count - jnp.bincount(indices_to_decs.reshape(-1), length=x.shape[0])
        dominate_count = dominate_count - next_front

        current_rank += 1
        next_front = dominate_count == 0
        return rank, dominate_count, current_rank, next_front

    rank, _dominate_count, _current_rank, _next_front = jax.lax.while_loop(
        _cond_fun,
        _body_fun,
        (rank, dominate_count, current_rank, next_front)
    )

    return rank
