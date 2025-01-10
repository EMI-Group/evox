import itertools
from math import comb
from typing import Tuple

import torch


def uniform_sampling(n: int, m: int) -> Tuple[torch.Tensor, int]:
    """Uniform sampling using Das and Dennis's method, Deb and Jain's method.
    Inspired by PlatEMO's NBI algorithm.

    :param n: Number of points to generate.
    :param m: Dimensionality of the grid.

    :return: The generated points, and the number of samples.
    """
    h1 = 1
    while comb(h1 + m, m - 1) <= n:
        h1 += 1

    # Generate combinations and scale them
    w = (
        torch.tensor(list(itertools.combinations(range(1, h1 + m), m - 1)))
        - torch.tile(torch.tensor(range(m - 1)), (comb(h1 + m - 1, m - 1), 1))
        - 1
    )
    w = (
        torch.cat([w, torch.zeros((w.size(0), 1), dtype=w.dtype) + h1], dim=1)
        - torch.cat([torch.zeros((w.size(0), 1), dtype=w.dtype), w], dim=1)
    ) / h1

    if h1 < m:
        h2 = 0
        while comb(h1 + m - 1, m - 1) + comb(h2 + m, m - 1) <= n:
            h2 += 1
        if h2 > 0:
            w2 = (
                torch.tensor(list(itertools.combinations(range(1, h2 + m), m - 1)))
                - torch.tile(torch.tensor(range(m - 1)), (comb(h2 + m - 1, m - 1), 1))
                - 1
            )
            w2 = (
                torch.cat([w2, torch.zeros((w2.size(0), 1), dtype=w2.dtype) + h2], dim=1)
                - torch.cat([torch.zeros((w2.size(0), 1), dtype=w2.dtype), w2], dim=1)
            ) / h2

            w = torch.cat([w, w2 / 2.0 + 1.0 / (2.0 * m)], dim=0)

    w = torch.maximum(w, torch.tensor(1e-6))
    n_samples = w.size(0)
    return w, n_samples
