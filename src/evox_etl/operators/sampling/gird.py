from math import ceil
from typing import Tuple

import evoxc


def grid_sampling(n: int, m: int) -> Tuple[evoxc.Tensor, int]:
    """Grid sampling.
    Inspired by PlatEMO.

    :param n: Number of grid points along each axis.
    :param m: Dimensionality of the grid (number of axes).

    :return: Grid points, and the number of samples.
    """
    num_points = int(ceil(n ** (1 / m)))

    # Generate grid points
    gap = evoxc.linspace(0.0, 1.0, num_points, dtype="float32")

    # Build the meshgrid axes via per-axis reshape + broadcast (evoxc has no
    # meshgrid), then stack them along the last axis.
    axes = []
    for i in range(m):
        shape = tuple([1] * i + [num_points] + [1] * (m - 1 - i))
        axes.append(evoxc.broadcast(evoxc.reshape(gap, shape), (num_points,) * m))

    # Stack grids along the last axis (axis=-1)
    w = evoxc.reshape(evoxc.stack(axes, axis=-1), (-1, m))

    # Reverse the order of columns to match JAX's `w[:, ::-1]`
    w = evoxc.flip(w, axes=1)

    num_samples = w.shape[0]
    return w, num_samples
