import torch
import itertools
from math import comb


def uniform_sampling(n: int, m: int):
    """
    Uniform sampling using Das and Dennis's method, Deb and Jain's method.
    Inspired by PlatEMO's NBI algorithm.
    Args:
        n: Number of points to generate.
        m: Dimensionality of the grid.
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
        torch.cat([w, torch.zeros((w.shape[0], 1), dtype=w.dtype) + h1], dim=1)
        - torch.cat([torch.zeros((w.shape[0], 1), dtype=w.dtype), w], dim=1)
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
                torch.cat(
                    [w2, torch.zeros((w2.shape[0], 1), dtype=w2.dtype) + h2], dim=1
                )
                - torch.cat([torch.zeros((w2.shape[0], 1), dtype=w2.dtype), w2], dim=1)
            ) / h2

            w = torch.cat([w, w2 / 2.0 + 1.0 / (2.0 * m)], dim=0)

    w = torch.maximum(w, torch.tensor(1e-6))
    n_samples = w.shape[0]
    return w, n_samples
