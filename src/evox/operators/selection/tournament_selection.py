from typing import List

import torch

from ...utils import lexsort


def tournament_selection_multifit(n_round: int, tournament_size: int, fitnesses: List[torch.Tensor]) -> torch.Tensor:
    """
    Perform tournament selection based on multiple fitness values.

    :param n_round: Number of rounds of selection (how many solutions to select).
    :type n_round: int
    :param tournament_size: Number of candidates in each tournament.
    :type tournament_size: int
    :param fitnesses: A list of 1D tensors representing the fitness values of candidates.
    :type fitnesses: list of torch.Tensor
    :return: Indices of the selected solutions after tournament selection.
    :rtype: torch.Tensor

    This function performs tournament selection by randomly selecting a group of candidates for each round,
    and selecting the best one from each group based on their fitness values.
    The fitnesses are compared lexicographically when multiple fitness values are provided.
    """
    # Combine fitness values into a single 2D tensor
    fitness_tensor = torch.stack(fitnesses, dim=1)

    # Get the number of candidates
    num_candidates = fitness_tensor.size(0)
    torch.manual_seed(1)
    # Perform tournament selection
    parents = torch.randint(0, num_candidates, (n_round, tournament_size))
    print(parents)

    # Gather fitness values for the selected candidates
    candidates_fitness = fitness_tensor[parents, ...]

    # Perform lexicographical sorting of fitness values to select the best candidate
    candidates_fitness = torch.stack([lexsort(candidates_fitness[i, :, :].T)
                                      for i in range(candidates_fitness.size(0))])

    # Select the best candidate from each tournament round
    selected_parents = torch.gather(parents, 1, candidates_fitness[:, 0].unsqueeze(1)).squeeze()

    return selected_parents


def tournament_selection(n_round: torch.Tensor, tournament_size: torch.Tensor, fitnesses: torch.Tensor) -> torch.Tensor:
    """
    Perform tournament selection based on a single fitness value.

    :param n_round: Number of rounds of selection (how many solutions to select).
    :type n_round: int
    :param tournament_size: Number of candidates in each tournament.
    :type tournament_size: int
    :param fitnesses: A 1D tensor representing the fitness values of candidates.
    :type fitnesses: torch.Tensor
    :return: Indices of the selected solutions after tournament selection.
    :rtype: torch.Tensor

    This function performs tournament selection by randomly selecting a group of candidates for each round,
    and selecting the best one from each group based on their fitness values.
    """
    # Get the number of candidates
    num_candidates = fitnesses.size(0)
    # Perform tournament selection
    parents = torch.randint(0, num_candidates, (n_round, tournament_size))
    print(parents)
    # Gather fitness values for the selected candidates
    candidates_fitness = fitnesses[parents]

    # Perform single fitness value selection
    winner_indices = torch.argmin(candidates_fitness, dim=1)

    # Select the best candidate from each tournament round
    # selected_parents = parents[torch.arange(n_round), winner_indices]
    selected_parents = torch.gather(parents, 1, winner_indices.unsqueeze(1)).squeeze()

    return selected_parents
