from jax import lax, jit
from evox import jit_class
from evox.operators.non_dominated_sort import non_dominated_sort
from functools import partial


@partial(jit, static_argnums=[2])
def non_dominate(population, fitness, topk):
    """Selection the topk individuals besed on their ranking with non-dominated sort,
    returns the selected population and the corresponding fitness.
    """
    ranking = non_dominated_sort(fitness)
    _, index = lax.topk(ranking, topk)
    return population[index], fitness[index]


@jit_class
class NonDominate:
    def __init__(self, topk):
        self.topk = topk

    def __call__(self, population, fitness):
        return non_dominate(population, fitness, self.topk)
