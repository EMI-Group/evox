from typing import List

import torch

from evox.utils import lexsort


def tournament_selection_multifit(n_round: int, fitnesses: List[torch.Tensor], tournament_size: int = 2) -> torch.Tensor:
    """
    Perform tournament selection based on multiple fitness values.

    :param n_round: Number of rounds of selection (how many solutions to select).
    :param fitnesses: A list of 1D tensors, each representing the fitness values of candidates for different objectives.
    :param tournament_size: Number of candidates in each tournament. Defaults to 2.
    :return: Indices of the selected solutions after tournament selection.

    This function performs tournament selection by randomly selecting a group of candidates for each round,
    and selecting the best one from each group based on their fitness values across multiple objectives.
    """
    fitness_tensor = torch.stack(fitnesses, dim=1)

    num_candidates = fitness_tensor.size(0)
    parents = torch.randint(0, num_candidates, (n_round, tournament_size), device=fitnesses[0].device)
    candidates_fitness = fitness_tensor[parents]
    candidates_fitness = lexsort(candidates_fitness.unbind(-1))

    selected_parents = torch.gather(parents, 1, candidates_fitness[:, 0].unsqueeze(1)).squeeze()

    return selected_parents


def tournament_selection(n_round: int, fitness: torch.Tensor, tournament_size: int = 2) -> torch.Tensor:
    """
    Perform tournament selection based on single fitness values.

    :param n_round: Number of rounds of selection (how many solutions to select).
    :param fitness: A 1D tensor representing the fitness values of candidates.
    :param tournament_size: Number of candidates in each tournament. Defaults to 2.
    :return: Indices of the selected solutions after tournament selection.

    This function performs tournament selection by randomly selecting a group of candidates for each round,
    and selecting the best one from each group based on their fitness values.
    """

    num_candidates = fitness.size(0)

    parents = torch.randint(0, num_candidates, (n_round, tournament_size), device=fitness.device)
    candidates_fitness = fitness[parents]

    winner_indices = torch.argmin(candidates_fitness, dim=1)

    selected_parents = torch.gather(parents, 1, winner_indices.unsqueeze(1)).squeeze()

    return selected_parents
