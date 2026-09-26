import itertools
from math import comb
from typing import Tuple

import etl


def uniform_sampling(n: int, m: int) -> Tuple[etl.Tensor, int]:
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
    combos = [list(c) for c in itertools.combinations(range(1, h1 + m), m - 1)]
    r = len(combos)
    w = (
        etl.constant(etl.core.tensor(combos, dtype="int64"))
        - etl.tile(
            etl.constant(etl.core.tensor(list(range(m - 1)), dtype="int64")), (r, 1)
        )
        - 1
    )
    w = (
        etl.concatenate(
            [
                w,
                etl.broadcast(
                    etl.constant(etl.core.tensor(h1, dtype="int64")), (r, 1)
                ),
            ],
            axis=1,
        )
        - etl.concatenate(
            [
                etl.broadcast(
                    etl.constant(etl.core.tensor(0, dtype="int64")), (r, 1)
                ),
                w,
            ],
            axis=1,
        )
    )
    w = etl.cast(w, "float32") / h1

    if h1 < m:
        h2 = 0
        while comb(h1 + m - 1, m - 1) + comb(h2 + m, m - 1) <= n:
            h2 += 1
        if h2 > 0:
            combos2 = [list(c) for c in itertools.combinations(range(1, h2 + m), m - 1)]
            r2 = len(combos2)
            w2 = (
                etl.constant(etl.core.tensor(combos2, dtype="int64"))
                - etl.tile(
                    etl.constant(etl.core.tensor(list(range(m - 1)), dtype="int64")),
                    (r2, 1),
                )
                - 1
            )
            w2 = (
                etl.concatenate(
                    [
                        w2,
                        etl.broadcast(
                            etl.constant(etl.core.tensor(h2, dtype="int64")), (r2, 1)
                        ),
                    ],
                    axis=1,
                )
                - etl.concatenate(
                    [
                        etl.broadcast(
                            etl.constant(etl.core.tensor(0, dtype="int64")), (r2, 1)
                        ),
                        w2,
                    ],
                    axis=1,
                )
            )
            w2 = etl.cast(w2, "float32") / h2

            w = etl.concatenate([w, w2 / 2.0 + 1.0 / (2.0 * m)], axis=0)

    w = etl.maximum(w, etl.constant(etl.core.tensor(1e-6, dtype="float32")))
    n_samples = w.shape[0]
    return w, n_samples
