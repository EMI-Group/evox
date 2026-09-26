"""Random personal-best selection (PSO variants)."""

import etl
import etl.random as random


def select_rand_pbest(key, percent: float, population, fitness):
    """
    Selects a random personal-best vector from the population for each individual.

    :param key: A PRNG key for random selection.
    :param percent: The proportion of the population to consider as best. Must be between 0 and 1.
    :param population: The population tensor of shape `(pop_size, dim)`.
    :param fitness: The fitness tensor of shape `(pop_size,)`.

    :return: A tensor containing the selected personal-best vector for each individual.
    """
    pop_size = population.shape[0]
    top_p_num = max(int(pop_size * percent), 1)
    pbest_indices_pool = etl.argsort(fitness)[:top_p_num]
    random_indices = random.randint(key, (pop_size,), 0, top_p_num, 'int32')
    return etl.gather(population, etl.gather(pbest_indices_pool, random_indices, axis=0), axis=0)
