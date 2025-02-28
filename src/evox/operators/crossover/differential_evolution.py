from typing import Tuple

import torch

from evox.utils import minimum_int


def DE_differential_sum(
    diff_padding_num: int, num_diff_vectors: torch.Tensor, index: torch.Tensor, population: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the difference vectors' sum in differential evolution.

    :param diff_padding_num: The number of padding difference vectors.
    :param num_diff_vectors: The number of difference vectors used in mutation.
    :param index: The index of current individual.
    :param population: The population tensor.

    :return: The difference sum and the index of first difference vector.
    """
    device = population.device
    pop_size = population.size(0)
    if num_diff_vectors.ndim == 0:
        num_diff_vectors = num_diff_vectors.unsqueeze(0)

    select_len = num_diff_vectors.unsqueeze(1) * 2 + 1
    rand_indices = torch.randint(0, pop_size, (pop_size, diff_padding_num), device=device)
    rand_indices = torch.where(rand_indices == index.unsqueeze(1), pop_size - 1, rand_indices)

    pop_permute = population[rand_indices]
    mask = torch.arange(diff_padding_num, device=device).unsqueeze(0) < select_len
    pop_permute_padding = torch.where(mask.unsqueeze(2), pop_permute, 0)

    diff_vectors = pop_permute_padding[:, 1:]
    difference_sum = diff_vectors[:, 0::2].sum(dim=1) - diff_vectors[:, 1::2].sum(dim=1)
    return difference_sum, rand_indices[:, 0]


def DE_binary_crossover(mutation_vector: torch.Tensor, current_vector: torch.Tensor, CR: torch.Tensor):
    """
    Performs binary crossover in differential evolution.

    :param mutation_vector: The mutated vector for each individual in the population.
    :param current_vector: The current vector for each individual in the population.
    :param CR: The crossover probability for each individual.

    :return: The trial vector after crossover for each individual.
    """
    device = mutation_vector.device
    pop_size, dim = mutation_vector.size()
    if CR.ndim == 1:
        CR = CR.unsqueeze(1)
    mask = torch.randn(pop_size, dim, device=device) < CR
    rind = torch.randint(0, dim, (pop_size,), device=device).unsqueeze(1)
    jind = torch.arange(dim, device=device).unsqueeze(0) == rind
    trial_vector = torch.where(torch.logical_or(mask, jind), mutation_vector, current_vector)
    return trial_vector


def DE_exponential_crossover(mutation_vector: torch.Tensor, current_vector: torch.Tensor, CR: torch.Tensor):
    """
    Performs exponential crossover in differential evolution.

    :param mutation_vector: The mutated vector for each individual in the population.
    :param current_vector: The current vector for each individual in the population.
    :param CR: The crossover probability for each individual.

    :return: The trial vector after crossover for each individual.
    """
    device = mutation_vector.device
    pop_size, dim = mutation_vector.size()
    nn = torch.randint(0, dim, (pop_size,), device=device)
    # Geometric distribution random ll
    float_tiny = 1.1754943508222875e-38
    ll = torch.rand(pop_size, device=device).clamp(min=float_tiny)
    ll = (ll.log() / (-CR.log1p())).floor().to(dtype=nn.dtype)
    mask = torch.arange(dim, device=device).unsqueeze(0) < (minimum_int(ll, dim) - 1).unsqueeze(1)
    mask = torch.gather(torch.tile(mask, (1, 2)), 1, nn.unsqueeze(1) + torch.arange(dim, device=device))
    trial_vector = torch.where(mask, mutation_vector, current_vector)
    return trial_vector


def DE_arithmetic_recombination(mutation_vector: torch.Tensor, current_vector: torch.Tensor, K: torch.Tensor):
    """
    Performs arithmetic recombination in differential evolution.

    :param mutation_vector: The mutated vector for each individual in the population.
    :param current_vector: The current vector for each individual in the population.
    :param K: The coefficient for each individual.

    :return: The trial vector after recombination for each individual.
    """
    if K.ndim == 1:
        K = K.unsqueeze(1)
    trial_vector = current_vector + K * (mutation_vector - current_vector)
    return trial_vector
