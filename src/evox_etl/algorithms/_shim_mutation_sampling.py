"""Mutation + sampling shims — functional ETL ports of torch evox operators.

Plain functions (no ``@etl.defn`` — call them inside an active trace).
1:1 ports of the read-only torch reference in
``src/evox/operators/mutation/pm_mutation.py`` and
``src/evox/operators/sampling/uniform.py``. numpy is used only at trace time
to bake static constants.
"""

import itertools
from math import comb
from typing import Tuple

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from etl import core


def polynomial_mutation(
    key: core.Tensor,
    x: core.Tensor,
    boundary: core.Tensor,
    pro_m: float = 1,
    dis_m: float = 20,
) -> core.Tensor:
    """Polynomial mutation, 1:1 port of ``evox.operators.mutation.pm_mutation``.

    ``boundary`` is a (2, d) tensor (row 0 = lower, row 1 = upper bounds);
    returns the mutated population (n x d).
    """
    lb = boundary[0]
    ub = boundary[1]
    n, d = x.shape
    k_site, k_mu = random.split_n(key, 2)
    site = random.uniform(k_site, (n, d), 0.0, 1.0) < (pro_m / d)
    mu = random.uniform(k_mu, (n, d), 0.0, 1.0)

    pop_dec = enp.maximum(enp.minimum(x, ub), lb)

    # Mutation for the first part where mu <= 0.5
    temp = enp.logical_and(site, mu <= 0.5)
    norm = etl.select(temp, (pop_dec - lb) / (ub - lb), 0.0)
    pop_dec = etl.select(
        temp,
        pop_dec
        + (ub - lb)
        * (
            enp.power(
                2 * mu + (1 - 2 * mu) * enp.power(1 - norm, dis_m + 1),
                1 / (dis_m + 1),
            )
            - 1
        ),
        pop_dec,
    )

    # Mutation for the second part where mu > 0.5
    temp = enp.logical_and(site, mu > 0.5)
    norm = etl.select(
        temp,
        (ub - pop_dec) / (ub - lb),
        enp.zeros(x.shape, dtype=x.dtype),
    )
    pop_dec = etl.select(
        temp,
        pop_dec
        + (ub - lb)
        * (
            1
            - enp.power(
                2 * (1 - mu) + 2 * (mu - 0.5) * enp.power(1 - norm, dis_m + 1),
                1 / (dis_m + 1),
            )
        ),
        pop_dec,
    )

    return pop_dec


def uniform_sampling(n: int, m: int) -> Tuple[core.Tensor, int]:
    """Uniform sampling via Das and Dennis's method, 1:1 port of
    ``evox.operators.sampling.uniform``.

    All itertools/comb math runs on Python ints at trace time; returns the
    point matrix baked as a constant and the sample count as a static int.
    """
    h1 = 1
    while comb(h1 + m, m - 1) <= n:
        h1 += 1

    def _layer(h: int) -> np.ndarray:
        """One Das-Dennis layer (n_samples = comb(h + m - 1, m - 1))."""
        rows = []
        for c in itertools.combinations(range(1, h + m), m - 1):
            v = [c[j] - j - 1 for j in range(m - 1)]
            row = [v[0]] + [v[j] - v[j - 1] for j in range(1, m - 1)]
            row.append(h - v[m - 2])
            rows.append(row)
        return np.asarray(rows, dtype=np.float32) / h

    w = _layer(h1)

    if h1 < m:
        h2 = 0
        while comb(h1 + m - 1, m - 1) + comb(h2 + m, m - 1) <= n:
            h2 += 1
        if h2 > 0:
            w = np.concatenate([w, _layer(h2) / 2.0 + 1.0 / (2.0 * m)], axis=0)

    n_samples = w.shape[0]
    points = enp.maximum(etl.ops.constant(core.tensor(w)), 1e-6)
    return points, n_samples
