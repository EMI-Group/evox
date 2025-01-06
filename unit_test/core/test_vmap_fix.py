import unittest

import torch

from evox.core import jit, vmap


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


class TestVmapFix(unittest.TestCase):
    def setUp(self):
        self.costs = torch.tensor(
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
        self.mask = torch.tensor([1, 0, 1, 0, 1, 1, 0], dtype=torch.bool)

    def test_distance_fn_with_mask(self):
        distances = jit(
            distance_fn, trace=True, lazy=False, example_inputs=(self.costs, self.mask)
        )
        self.assertIsNotNone(distances(self.costs, self.mask))
        self.assertIsNotNone(distances(self.costs[:-1], self.mask[:-1]))

    def test_distance_fn_without_mask(self):
        distances = jit(
            distance_fn, trace=True, lazy=False, example_inputs=(self.costs,)
        )
        self.assertIsNotNone(distances(self.costs))

    def test_distance_fn_with_none(self):
        distances = jit(
            distance_fn, trace=True, lazy=False, example_inputs=(self.costs, None)
        )
        self.assertIsNotNone(distances(self.costs, None))
