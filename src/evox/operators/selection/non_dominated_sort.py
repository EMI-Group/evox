import jax
import jax.numpy as jnp
from jax import vmap, lax
import chex
from evox.utils import _dominate_relation


@jax.jit
def non_dominated_sort(x, method="auto"):
    """Perform non-dominated sort
    Parameters
    ----------
    x
        An array with shape (n, m) where n is the population size, m is the number of objectives
    method
        Determine the jax operation used.
        Default to "scan" on CPU and "full map-reduce" on GPU.

    Returns
    -------
    jax.Array
        A one-dimensional array representing the ranking, starts with 0.
    """
    assert method in [
        "auto",
        "scan",
        "full map-reduce",
    ], "method must be either 'auto', 'scan', or 'full map-reduce'"
    if method == "auto":
        backend = jax.default_backend()
        if backend == "cpu":
            method = "scan"
        else:
            method = "full map-reduce"

    chex.assert_rank(x, 2)
    dominate_relation_matrix = _dominate_relation(x, x)
    rank = jnp.zeros((x.shape[0],), dtype=jnp.int32)
    dominate_count = jnp.sum(dominate_relation_matrix, axis=0)

    current_rank = 0
    pareto_front = dominate_count == 0

    def _cond_fun(loop_state):
        _rank, _dominate_count, _current_rank, pareto_front = loop_state
        return jnp.any(pareto_front)

    def _body_fun(loop_state):
        rank, dominate_count, current_rank, pareto_front = loop_state
        rank = jnp.where(pareto_front, current_rank, rank)  # update rank

        def dominate_relation_or_zero(count_desc, dominate_relation_and_cond):
            dominate_relation, cond = dominate_relation_and_cond
            count_desc = lax.cond(
                cond, lambda count: count + dominate_relation, lambda x: x, count_desc
            )
            return count_desc, None

        if method == "full map-reduce":
            count_desc = jnp.sum(
                pareto_front[:, jnp.newaxis] * dominate_relation_matrix, axis=0
            )
        else:
            count_desc, _ = lax.scan(
                dominate_relation_or_zero,
                jnp.zeros_like(dominate_count),
                (dominate_relation_matrix, pareto_front),
            )
        # a trick to prevent the current pareto-front from being selected again
        dominate_count = dominate_count - count_desc
        dominate_count = dominate_count - pareto_front

        current_rank += 1
        pareto_front = dominate_count == 0
        return rank, dominate_count, current_rank, pareto_front

    rank, _dominate_count, _current_rank, _pareto_front = jax.lax.while_loop(
        _cond_fun, _body_fun, (rank, dominate_count, current_rank, pareto_front)
    )

    return rank
