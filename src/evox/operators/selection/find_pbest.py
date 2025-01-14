import torch


def select_rand_pbest(percent: float, population: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
    """
    Selects a random personal-best vector from the population for each individual.

    :param percent: The proportion of the population to consider as best. Must be between 0 and 1.
    :param population: The population tensor of shape `(pop_size, dim)`.
    :param fitness: The fitness tensor of shape `(pop_size,)`.

    :return: A tensor containing the selected personal-best vector for each individual.
    """
    device = population.device
    pop_size = population.size(0)
    top_p_num = max(int(pop_size * percent), 1)
    pbest_indices_pool = torch.argsort(fitness)[:top_p_num]
    random_indices = torch.randint(0, top_p_num, (pop_size,), device=device)
    return population[pbest_indices_pool[random_indices]]
