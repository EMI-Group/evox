from jax import lax, jit
from evox import jit_class
from functools import partial


@partial(jit, static_argnums=[2])
def topk_fit(population, fitness, topk):
    """Selection the topk individuals besed on the fitness,
    returns the selected population and the corresponding fitness.
    """
    topk_fit, index = lax.top_k(-fitness, topk)
    return population[index], -topk_fit


@jit_class
class TopkFit:
    def __init__(self, topk):
        self.topk = topk

    def __call__(self, population, fitness):
        return topk_fit(population, fitness, self.topk)
