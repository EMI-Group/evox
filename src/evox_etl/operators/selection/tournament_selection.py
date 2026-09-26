"""Tournament selection operators (single- and multi-fitness)."""

from typing import List

import etl
import etl.numpy as enp
import etl.random as random

from .non_dominate import _lexsort, _take_along_axis


def tournament_selection(key, n_round: int, fitness, tournament_size: int = 2):
    """
    Perform tournament selection based on single fitness values.

    :param key: A PRNG key for random candidate sampling.
    :param n_round: Number of rounds of selection (how many solutions to select).
    :param fitness: A 1D tensor representing the fitness values of candidates.
    :param tournament_size: Number of candidates in each tournament. Defaults to 2.
    :return: Indices of the selected solutions after tournament selection.

    This function performs tournament selection by randomly selecting a group of candidates for each round,
    and selecting the best one from each group based on their fitness values.
    """

    num_candidates = fitness.shape[0]

    parents = random.randint(key, (n_round, tournament_size), 0, num_candidates, 'int32')
    candidates_fitness = etl.gather(fitness, parents, axis=0)

    winner_indices = etl.argmin(candidates_fitness, axis=1)

    selected_parents = etl.gather(
        etl.reshape(parents, (-1,)),
        enp.arange(n_round, dtype='int64') * tournament_size + etl.cast(winner_indices, 'int64'),
    )

    return selected_parents


def tournament_selection_multifit(
    key, n_round: int, fitnesses: List, tournament_size: int = 2
):
    """
    Perform tournament selection based on multiple fitness values.

    :param key: A PRNG key for random candidate sampling.
    :param n_round: Number of rounds of selection (how many solutions to select).
    :param fitnesses: A list of 1D tensors, each representing the fitness values of candidates for different objectives.
    :param tournament_size: Number of candidates in each tournament. Defaults to 2.
    :return: Indices of the selected solutions after tournament selection.

    This function performs tournament selection by randomly selecting a group of candidates for each round,
    and selecting the best one from each group based on their fitness values across multiple objectives.
    """
    fitness_tensor = etl.stack(fitnesses, axis=1)
    n_fit = fitness_tensor.shape[1]

    num_candidates = fitness_tensor.shape[0]
    parents = random.randint(key, (n_round, tournament_size), 0, num_candidates, 'int32')
    candidates_fitness = etl.gather(fitness_tensor, parents, axis=0)
    keys = [
        etl.gather(candidates_fitness, etl.constant(etl.core.tensor(i, dtype='int64')), axis=-1)
        for i in range(n_fit)
    ]
    candidates_order = _lexsort(keys, axis=1)

    selected_parents = _take_along_axis(parents, candidates_order[:, :1], axis=1)
    return etl.reshape(selected_parents, (n_round,))
