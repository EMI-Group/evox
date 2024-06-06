from jax import lax, jit
import jax.numpy as jnp
from evox import jit_class
from functools import partial


@partial(jit, static_argnums=[2, 3])
def topk_fit(population, fitness, topk, deduplicate):
    """Selection the topk individuals besed on the fitness,
    returns the selected population and the corresponding fitness.
    """

    if deduplicate:
        # remove duplicated individuals by assigning their fitness to inf
        _, unique_index, unique_count = jnp.unique(
            population,
            axis=0,
            size=population.shape[0],
            return_index=True,
            return_counts=True,
        )
        population = population[unique_index]
        fitness = fitness[unique_index]
        count = jnp.sum(unique_count > 0)
        # backup the original fitness
        # so even when a duplicated individual is selected, the original fitness is used
        # this will happen if the topk is larger than the number of unique individuals
        fitness_bak = fitness
        fitness = jnp.where(jnp.arange(fitness.shape[0]) < count, fitness, jnp.inf)

    index = jnp.argsort(fitness)
    index = index[:topk]

    if deduplicate:
        return population[index], fitness_bak[index]
    else:
        return population[index], fitness[index]


@jit_class
class TopkFit:
    def __init__(self, topk, deduplicate=False):
        self.topk = topk
        self.deduplicate = deduplicate

    def __call__(self, population, fitness):
        return topk_fit(population, fitness, self.topk, self.deduplicate)
