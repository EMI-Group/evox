from math import ceil
from typing import Tuple

import torch


def grid_sampling(n: int, m: int) -> Tuple[torch.Tensor, int]:
    """Grid sampling.
    Inspired by PlatEMO.

    :param n: Number of grid points along each axis.
    :param m: Dimensionality of the grid (number of axes).

    :return: Grid points, and the number of samples.
    """
    num_points = int(ceil(n ** (1 / m)))

    # Generate grid points
    gap = torch.linspace(0, 1, num_points)
    grid_axes = [gap for _ in range(m)]

    # Generate grid using meshgrid and stack values
    grid_values = torch.meshgrid(*grid_axes, indexing="ij")

    # Stack grids along the last axis (axis=-1)
    w = torch.stack(grid_values, dim=-1).reshape(-1, m)

    # Reverse the order of columns to match JAX's `w[:, ::-1]`
    w = w.flip(dims=[1])

    num_samples = w.size(0)
    return w, num_samples
