from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, pure_callback, vmap

from evox import jit_class
from evox.utils import dominate_relation


def host_rank_from_domination_matrix(dominate_mat, dominate_count):
    dominate_count = np.copy(dominate_count)
    # The number of inidividuals
    N = dominate_mat.shape[0]
    rank = np.empty((N,), dtype=np.int32)
    current_rank = 0
    pareto_front = dominate_count == 0
    while pareto_front.any():
        rank[pareto_front] = current_rank  # update rank
        count_desc = np.sum(dominate_mat[pareto_front, :], axis=0)
        # a trick to prevent the current pareto-front from being selected again
        dominate_count -= count_desc
        dominate_count -= pareto_front
        current_rank += 1
        pareto_front = dominate_count == 0

    return rank


@jit
def non_dominated_sort(x, method="auto"):
    """Perform non-dominated sort
    Parameters
    ----------
    x
        An array with shape (n, m) where n is the population size, m is the number of objectives
    method
        Determine the jax operation used.
        Default to "scan" on CPU and "full map-reduce" on GPU.
        An experimental "host" mode can be used for CPU, to run computation on CPU through host callback.

    Returns
    -------
    jax.Array
        A one-dimensional array representing the ranking, starts with 0.
    """
    assert method in [
        "auto",
        "scan",
        "full map-reduce",
        "host",
    ], "method must be either 'auto', 'scan', or 'full map-reduce', 'host'"
    if method == "auto":
        backend = jax.default_backend()
        if backend == "cpu":
            method = "scan"
        else:
            method = "full map-reduce"

    dominate_relation_matrix = dominate_relation(x, x)
    dominate_count = jnp.sum(dominate_relation_matrix, axis=0)

    if method == "host":
        rank = pure_callback(
            host_rank_from_domination_matrix,
            jax.ShapeDtypeStruct((x.shape[0],), dtype=jnp.int32),
            dominate_relation_matrix,
            dominate_count,
        )
        return rank

    rank = jnp.zeros((x.shape[0],), dtype=jnp.int32)
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


@jit
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
        distance_range = cost[num_valid_elem - 1] - cost[0]
        distance = jnp.empty(totel_len)
        distance = distance.at[rank[1:-1]].set((cost[2:] - cost[:-2]) / distance_range)
        distance = distance.at[rank[0]].set(jnp.inf)
        distance = distance.at[rank[num_valid_elem - 1]].set(jnp.inf)

        distance = jnp.where(mask, distance, -jnp.inf)
        return distance

    return jnp.sum(jax.vmap(distance_in_one_dim, 1, 1)(costs), axis=1)


@jit
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


@partial(jit, static_argnums=[2])
def non_dominate(population, fitness, topk):
    """Selection the topk individuals besed on their ranking with non-dominated sort,
    returns the selected population and the corresponding fitness.
    """
    # first apply non_dominated sort
    rank = non_dominated_sort(fitness)
    # then find the worst rank within topk, and use crodwing_distance_sort as tiebreaker
    order = jnp.argsort(rank)
    worst_rank = rank[order[topk-1]]
    mask = rank == worst_rank
    crowding_distance = crowding_distance_sort(fitness, mask)

    combined_order = jnp.lexsort((-crowding_distance, rank))[:topk]
    return population[combined_order], fitness[combined_order]


@jit_class
class NonDominate:
    def __init__(self, topk):
        self.topk = topk

    def __call__(self, population, fitness):
        return non_dominate(population, fitness, self.topk)
