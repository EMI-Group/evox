from typing import Tuple

import torch


def sort_by_key(keys: torch.Tensor, population: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    sorted_population = population[order]
    return sorted_keys, sorted_population
