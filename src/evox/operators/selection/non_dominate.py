from jax import lax, jit
import jax.numpy as jnp
from evox import jit_class
from evox.operators.non_dominated_sort import non_dominated_sort
from evox.operators.crowding_distance_sort import crowding_distance_sort
from functools import partial


@partial(jit, static_argnums=[2])
def non_dominate(population, fitness, topk):
    """Selection the topk individuals besed on their ranking with non-dominated sort,
    returns the selected population and the corresponding fitness.
    """
    # first apply non_dominated sort
    rank = non_dominated_sort(fitness)
    # then find the worst rank within topk, and use crodwing_distance_sort as tiebreaker
    worst_rank = -lax.top_k(-rank, topk)
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
