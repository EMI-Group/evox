import torch

import os
import sys

current_directory = os.getcwd()
if current_directory not in sys.path:
    sys.path.append(current_directory)

from src.core import vmap, jit


def distance_fn(costs: torch.Tensor, mask: torch.Tensor = None):
    total_len = costs.shape[0]
    if mask is None:
        num_valid_elem = total_len
        mask = torch.ones(total_len, dtype=torch.bool, device=costs.device)
    else:
        num_valid_elem = mask.sum()

    def distance_in_one_dim(cost: torch.Tensor):
        rank = torch.argsort(cost * mask)
        cost = cost[rank]
        distance_range = cost[num_valid_elem - 1] - cost[0]
        distance = torch.empty_like(cost)
        distance[rank[1:-1]] = (cost[2:] - cost[:-2]) / distance_range[None]
        distance[rank[0]] = float("inf")
        distance[rank[num_valid_elem - 1]] = float("inf")
        distance = torch.where(mask, distance, float("-inf"))
        return distance

    return vmap(distance_in_one_dim, in_dims=1, out_dims=1, trace=False)(costs)


if __name__ == "__main__":
    costs = torch.tensor(
        [
            [1.0, 0.8, 0.5],
            [2.0, 1.0, 3.0],
            [3.0, 3.0, 3.0],
            [4.0, 1.0, 2.0],
            [1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5],
            [1.5, 4.5, 3.5],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([1, 0, 1, 0, 1, 1, 0], dtype=torch.bool)
    distances = jit(distance_fn, trace=True, lazy=False, example_inputs=(costs, mask))
    print(distances(costs, mask))
    print(distances(costs[:-1], mask[:-1]))
