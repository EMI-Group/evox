"""DTLZ test suite (DTLZ1-7) — functional ETL port of torch evox
``src/evox/problems/numerical/dtlz.py`` (read-only reference; math mirrored
1:1, all ``torch.*`` ops replaced with ``etl`` / ``etl.numpy`` ops).

The torch ``DTLZ`` base class disappears: its ``sample`` (uniform sampling,
``ref_num * m`` rows) becomes the ``_sample`` helper, its ``pf`` (=
``sample / 2``) becomes DTLZ1's pf, and DTLZ3/DTLZ4 inherit DTLZ2's pf
exactly like the torch subclasses.

Notes for agents:
- ``evaluate`` / ``pf`` are module-level dispatchers keyed on config type
  (static ``isinstance`` at trace time — the config is a compile-time static
  arg). The per-variant functions are ``evaluate_dtlzN`` / ``pf_dtlzN`` and
  mirror each torch class method exactly.
- Traced shapes must be fully STATIC (batch ``n`` AND dim ``d``): etl's IR
  slice op cannot express full-axis slices over symbolic dims, and every
  DTLZ formula slices ``X`` on the feature axis (``X[:, m - 1:]``).
  ``StdWorkflow`` builds specs from concrete tensors, so this holds in
  practice; use static specs in tests.
- ``_uniform_sampling`` / ``_grid_sampling`` are PRIVATE local mirrors of
  torch ``operators/sampling/uniform.py`` / ``gird.py`` (same math, minus
  device and the ``(tensor, n_samples)`` tuple) — the operators milestone
  will move/dedupe them.
"""

__all__ = [
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
    "evaluate",
    "pf",
]

import itertools
from dataclasses import dataclass
from math import ceil, comb, pi
from typing import Any

import etl
import etl.numpy as enp


@dataclass(frozen=True)
class DTLZ1:
    """DTLZ1 config (fields mirror torch ``DTLZ1.__init__``; no device)."""

    d: int = 7
    m: int = 3
    ref_num: int = 1000


@dataclass(frozen=True)
class DTLZ2:
    """DTLZ2 config (fields mirror torch ``DTLZ2.__init__``; no device)."""

    d: int = 12
    m: int = 3
    ref_num: int = 1000


@dataclass(frozen=True)
class DTLZ3:
    """DTLZ3 config (fields mirror torch ``DTLZ3.__init__``; no device)."""

    d: int = 12
    m: int = 3
    ref_num: int = 1000


@dataclass(frozen=True)
class DTLZ4:
    """DTLZ4 config (fields mirror torch ``DTLZ4.__init__``; no device)."""

    d: int = 12
    m: int = 3
    ref_num: int = 1000


@dataclass(frozen=True)
class DTLZ5:
    """DTLZ5 config (fields mirror torch ``DTLZ5.__init__``; no device)."""

    d: int = 12
    m: int = 3
    ref_num: int = 1000


@dataclass(frozen=True)
class DTLZ6:
    """DTLZ6 config (fields mirror torch ``DTLZ6.__init__``; no device)."""

    d: int = 12
    m: int = 3
    ref_num: int = 1000


@dataclass(frozen=True)
class DTLZ7:
    """DTLZ7 config (fields mirror torch ``DTLZ7.__init__``; no device)."""

    d: int = 21
    m: int = 3
    ref_num: int = 1000


DTLZConfig = DTLZ1 | DTLZ2 | DTLZ3 | DTLZ4 | DTLZ5 | DTLZ6 | DTLZ7


def _uniform_sampling(n: int, m: int) -> etl.SymbolicTensor:
    """Das-Dennis uniform sampling; local 1:1 mirror of torch
    ``operators/sampling/uniform.py`` (operators milestone will dedupe)."""
    h1 = 1
    while comb(h1 + m, m - 1) <= n:
        h1 += 1

    c = comb(h1 + m - 1, m - 1)
    combos = list(itertools.combinations(range(1, h1 + m), m - 1))
    w = (
        etl.constant(etl.core.tensor(combos, dtype=etl.int64))
        - etl.tile(etl.constant(etl.core.tensor(range(m - 1), dtype=etl.int64)), (c, 1))
        - 1
    )
    w = (
        enp.concatenate([w, enp.zeros((c, 1), dtype=etl.int64) + h1], axis=1)
        - enp.concatenate([enp.zeros((c, 1), dtype=etl.int64), w], axis=1)
    )
    w = etl.cast(w, etl.float32) / h1

    if h1 < m:
        h2 = 0
        while comb(h1 + m - 1, m - 1) + comb(h2 + m, m - 1) <= n:
            h2 += 1
        if h2 > 0:
            c2 = comb(h2 + m - 1, m - 1)
            combos2 = list(itertools.combinations(range(1, h2 + m), m - 1))
            w2 = (
                etl.constant(etl.core.tensor(combos2, dtype=etl.int64))
                - etl.tile(
                    etl.constant(etl.core.tensor(range(m - 1), dtype=etl.int64)),
                    (c2, 1),
                )
                - 1
            )
            w2 = (
                enp.concatenate([w2, enp.zeros((c2, 1), dtype=etl.int64) + h2], axis=1)
                - enp.concatenate([enp.zeros((c2, 1), dtype=etl.int64), w2], axis=1)
            )
            w2 = etl.cast(w2, etl.float32) / h2

            w = enp.concatenate([w, w2 / 2.0 + 1.0 / (2.0 * m)], axis=0)

    w = enp.maximum(w, 1e-6)
    return w


def _grid_sampling(n: int, m: int) -> etl.SymbolicTensor:
    """Grid sampling; local 1:1 mirror of torch ``operators/sampling/gird.py``
    (operators milestone will dedupe)."""
    num_points = int(ceil(n ** (1 / m)))

    # torch: gap = torch.linspace(0, 1, num_points). etl.linspace is the 1:1
    # op translation; its backend kernels may differ from torch's linspace
    # kernel by ~1 float32 ulp on some entries (irrelevant: DTLZ7 pf parity
    # holds to ~1e-8).
    gap = etl.linspace(0.0, 1.0, num_points, dtype=etl.float32)

    # torch: torch.meshgrid(*grid_axes, indexing="ij") then stack along -1
    # and reshape(-1, m) — broadcast each axis along its own dim instead.
    shape = (num_points,) * m
    axes = []
    for k in range(m):
        shp = [1] * m
        shp[k] = num_points
        axes.append(enp.broadcast_to(enp.reshape(gap, tuple(shp)), shape))
    w = enp.reshape(etl.stack(axes, axis=-1), (-1, m))

    w = etl.flip(w, axes=[1])
    return w


def _sample(config: DTLZConfig) -> etl.SymbolicTensor:
    """Base DTLZ sample — torch ``DTLZ.__init__``:
    ``uniform_sampling(ref_num * m, m)``."""
    return _uniform_sampling(config.ref_num * config.m, config.m)


def evaluate_dtlz1(
    config: DTLZ1, problem_state: Any, X: etl.SymbolicTensor
) -> tuple[etl.SymbolicTensor, Any]:
    """DTLZ1 objectives (mirrors torch ``DTLZ1.evaluate``); problem_state
    passes through unchanged."""
    m = config.m
    n, d = X.shape
    g = 100 * (
        d
        - m
        + 1
        + enp.sum(
            (X[:, m - 1 :] - 0.5) ** 2 - enp.cos(20 * pi * (X[:, m - 1 :] - 0.5)),
            axis=1,
            keepdims=True,
        )
    )
    flip_cumprod = etl.flip(
        etl.cumprod(
            enp.concatenate([enp.ones((n, 1), dtype=X.dtype), X[:, : m - 1]], axis=1),
            axis=1,
        ),
        axes=[1],
    )
    rest_part = enp.concatenate(
        [
            enp.ones((n, 1), dtype=X.dtype),
            1 - etl.flip(X[:, : m - 1], axes=[1]),
        ],
        axis=1,
    )
    f = 0.5 * (1 + g) * flip_cumprod * rest_part
    return f, problem_state


def evaluate_dtlz2(
    config: DTLZ2, problem_state: Any, X: etl.SymbolicTensor
) -> tuple[etl.SymbolicTensor, Any]:
    """DTLZ2 objectives (mirrors torch ``DTLZ2.evaluate``); problem_state
    passes through unchanged."""
    m = config.m
    g = enp.sum((X[:, m - 1 :] - 0.5) ** 2, axis=1, keepdims=True)
    f = (
        (1 + g)
        * etl.flip(
            etl.cumprod(
                enp.concatenate(
                    [
                        enp.ones((X.shape[0], 1), dtype=X.dtype),
                        enp.maximum(enp.cos(X[:, : m - 1] * pi / 2), 0.0),
                    ],
                    axis=1,
                ),
                axis=1,
            ),
            axes=[1],
        )
        * enp.concatenate(
            [
                enp.ones((X.shape[0], 1), dtype=X.dtype),
                enp.sin(etl.flip(X[:, : m - 1], axes=[1]) * pi / 2),
            ],
            axis=1,
        )
    )
    return f, problem_state


def evaluate_dtlz3(
    config: DTLZ3, problem_state: Any, X: etl.SymbolicTensor
) -> tuple[etl.SymbolicTensor, Any]:
    """DTLZ3 objectives (mirrors torch ``DTLZ3.evaluate``); problem_state
    passes through unchanged."""
    n, d = X.shape
    m = config.m
    g = 100 * (
        d
        - m
        + 1
        + enp.sum(
            (X[:, m - 1 :] - 0.5) ** 2 - enp.cos(20 * pi * (X[:, m - 1 :] - 0.5)),
            axis=1,
            keepdims=True,
        )
    )
    f = (
        (1 + g)
        * etl.flip(
            etl.cumprod(
                enp.concatenate(
                    [
                        enp.ones((n, 1), dtype=X.dtype),
                        enp.maximum(enp.cos(X[:, : m - 1] * pi / 2), 0.0),
                    ],
                    axis=1,
                ),
                axis=1,
            ),
            axes=[1],
        )
        * enp.concatenate(
            [
                enp.ones((n, 1), dtype=X.dtype),
                enp.sin(etl.flip(X[:, : m - 1], axes=[1]) * pi / 2),
            ],
            axis=1,
        )
    )
    return f, problem_state


def evaluate_dtlz4(
    config: DTLZ4, problem_state: Any, X: etl.SymbolicTensor
) -> tuple[etl.SymbolicTensor, Any]:
    """DTLZ4 objectives (mirrors torch ``DTLZ4.evaluate``); problem_state
    passes through unchanged."""
    m = config.m

    Xfront = enp.power(X[:, : m - 1], 100)
    Xrear = X[:, m - 1 :]

    g = enp.sum((Xrear - 0.5) ** 2, axis=1, keepdims=True)

    f = (
        (1 + g)
        * etl.flip(
            etl.cumprod(
                enp.concatenate(
                    [
                        enp.ones((g.shape[0], 1), dtype=X.dtype),
                        enp.maximum(enp.cos(Xfront * pi / 2), 0.0),
                    ],
                    axis=1,
                ),
                axis=1,
            ),
            axes=[1],
        )
        * enp.concatenate(
            [
                enp.ones((g.shape[0], 1), dtype=X.dtype),
                enp.sin(etl.flip(Xfront, axes=[1]) * pi / 2),
            ],
            axis=1,
        )
    )
    return f, problem_state


def evaluate_dtlz5(
    config: DTLZ5, problem_state: Any, X: etl.SymbolicTensor
) -> tuple[etl.SymbolicTensor, Any]:
    """DTLZ5 objectives (mirrors torch ``DTLZ5.evaluate``); problem_state
    passes through unchanged."""
    m = config.m

    g = enp.sum((X[:, m - 1 :] - 0.5) ** 2, axis=1, keepdims=True)
    temp = etl.tile(g, (1, m - 2))

    # torch: Xfront = X[:, : m - 1].clone(); Xfront[:, 1:] = (1 + 2 * temp *
    # Xfront[:, 1:]) / (2 + 2 * temp) — composed immutably here.
    Xfront = enp.concatenate(
        [X[:, :1], (1 + 2 * temp * X[:, 1 : m - 1]) / (2 + 2 * temp)], axis=1
    )

    f = (
        (1 + g)
        * etl.flip(
            etl.cumprod(
                enp.concatenate(
                    [
                        enp.ones((g.shape[0], 1), dtype=X.dtype),
                        enp.maximum(enp.cos(Xfront * pi / 2), 0.0),
                    ],
                    axis=1,
                ),
                axis=1,
            ),
            axes=[1],
        )
        * enp.concatenate(
            [
                enp.ones((g.shape[0], 1), dtype=X.dtype),
                enp.sin(etl.flip(Xfront, axes=[1]) * pi / 2),
            ],
            axis=1,
        )
    )
    return f, problem_state


def evaluate_dtlz6(
    config: DTLZ6, problem_state: Any, X: etl.SymbolicTensor
) -> tuple[etl.SymbolicTensor, Any]:
    """DTLZ6 objectives (mirrors torch ``DTLZ6.evaluate``); problem_state
    passes through unchanged."""
    m = config.m
    g = enp.sum(X[:, m - 1 :] ** 0.1, axis=1, keepdims=True)
    temp = etl.tile(g, (1, m - 2))
    # Same immutably-composed Xfront update as DTLZ5.
    Xfront = enp.concatenate(
        [X[:, :1], (1 + 2 * temp * X[:, 1 : m - 1]) / (2 + 2 * temp)], axis=1
    )

    f = (
        etl.tile(1 + g, (1, m))
        * etl.flip(
            etl.cumprod(
                enp.concatenate(
                    [
                        enp.ones((X.shape[0], 1), dtype=X.dtype),
                        enp.maximum(enp.cos(Xfront * pi / 2), 0.0),
                    ],
                    axis=1,
                ),
                axis=1,
            ),
            axes=[1],
        )
        * enp.concatenate(
            [
                enp.ones((X.shape[0], 1), dtype=X.dtype),
                enp.sin(etl.flip(Xfront, axes=[1]) * pi / 2),
            ],
            axis=1,
        )
    )
    return f, problem_state


def evaluate_dtlz7(
    config: DTLZ7, problem_state: Any, X: etl.SymbolicTensor
) -> tuple[etl.SymbolicTensor, Any]:
    """DTLZ7 objectives (mirrors torch ``DTLZ7.evaluate``); problem_state
    passes through unchanged."""
    m = config.m
    g = 1 + 9 * enp.mean(X[:, m - 1 :], axis=1, keepdims=True)

    term = enp.sum(
        X[:, : m - 1]
        / (1 + etl.tile(g, (1, m - 1)))
        * (1 + enp.sin(3 * pi * X[:, : m - 1])),
        axis=1,
        keepdims=True,
    )
    f = enp.concatenate([X[:, : m - 1], (1 + g) * (m - term)], axis=1)
    return f, problem_state


def pf_dtlz1(config: DTLZ1) -> etl.SymbolicTensor:
    """DTLZ1 Pareto front: uniform sample / 2 (torch base ``DTLZ.pf``)."""
    return _sample(config) / 2


def pf_dtlz2(config: DTLZ2 | DTLZ3 | DTLZ4) -> etl.SymbolicTensor:
    """DTLZ2 Pareto front: row-normalized uniform samples — DTLZ3/DTLZ4
    inherit it in torch."""
    sample = _sample(config)
    return sample / enp.sqrt(enp.sum(sample ** 2, axis=1, keepdims=True))


def _pf_dtlz56(config: DTLZ5 | DTLZ6) -> etl.SymbolicTensor:
    """Shared DTLZ5/DTLZ6 Pareto front (identical bodies in torch)."""
    n = config.ref_num * config.m
    m = config.m

    # torch: vstack of two hstacks (arange rows + scalar), then .T
    f = etl.transpose(
        etl.stack(
            [
                enp.concatenate(
                    [
                        enp.arange(0.0, 1.0, 1.0 / (n - 1), dtype=etl.float32),
                        enp.ones((1,), dtype=etl.float32),
                    ],
                    axis=0,
                ),
                enp.concatenate(
                    [
                        enp.arange(1.0, 0.0, -1.0 / (n - 1), dtype=etl.float32),
                        enp.zeros((1,), dtype=etl.float32),
                    ],
                    axis=0,
                ),
            ],
            axis=0,
        )
    )

    f = f / etl.tile(
        enp.sqrt(enp.sum(f ** 2, axis=1, keepdims=True)), (1, f.shape[1])
    )

    for _ in range(m - 2):
        f = enp.concatenate([f[:, 0:1], f], axis=1)

    # torch: f / sqrt(2.0) ** tile(hstack([m - 2, arange(m - 2, -1, -1)]),
    # (f.size(0), 1)) — int exponents cast to float32 (etl promotes
    # float32 ** int64 to float64).
    f = f / enp.power(
        enp.sqrt(etl.constant(etl.core.tensor(2.0, dtype=etl.float32))),
        etl.tile(
            etl.cast(
                enp.concatenate(
                    [enp.arange(m - 2, m - 1), enp.arange(m - 2, -1, -1)], axis=0
                ),
                etl.float32,
            ),
            (f.shape[0], 1),
        ),
    )
    return f


def pf_dtlz5(config: DTLZ5) -> etl.SymbolicTensor:
    """DTLZ5 Pareto front (mirrors torch ``DTLZ5.pf``)."""
    return _pf_dtlz56(config)


def pf_dtlz6(config: DTLZ6) -> etl.SymbolicTensor:
    """DTLZ6 Pareto front (mirrors torch ``DTLZ6.pf``)."""
    return _pf_dtlz56(config)


def pf_dtlz7(config: DTLZ7) -> etl.SymbolicTensor:
    """DTLZ7 Pareto front (mirrors torch ``DTLZ7.pf``)."""
    m = config.m
    interval = etl.constant(
        etl.core.tensor([0.0, 0.251412, 0.631627, 0.859401], dtype=etl.float32)
    )
    median = (interval[1] - interval[0]) / (
        interval[3] - interval[2] + interval[1] - interval[0]
    )

    x = _grid_sampling(config.ref_num * config.m, m - 1)

    mask_less_equal_median = x <= median
    mask_greater_median = x > median

    x = enp.where(
        mask_less_equal_median,
        x * (interval[1] - interval[0]) / median + interval[0],
        x,
    )
    x = enp.where(
        mask_greater_median,
        (x - median) * (interval[3] - interval[2]) / (1 - median) + interval[2],
        x,
    )

    last_col = 2 * (
        m - enp.sum(x / 2 * (1 + enp.sin(3 * pi * x)), axis=1, keepdims=True)
    )

    return enp.concatenate([x, last_col], axis=1)


def evaluate(
    config: DTLZConfig, problem_state: Any, X: etl.SymbolicTensor
) -> tuple[etl.SymbolicTensor, Any]:
    """``evaluate(config, problem_state, pop) -> (fitness, problem_state)``.

    Routes to the DTLZ1-7 variant evaluator by config type (static
    ``isinstance`` — the config is a compile-time static arg).
    """
    if isinstance(config, DTLZ1):
        return evaluate_dtlz1(config, problem_state, X)
    if isinstance(config, DTLZ2):
        return evaluate_dtlz2(config, problem_state, X)
    if isinstance(config, DTLZ3):
        return evaluate_dtlz3(config, problem_state, X)
    if isinstance(config, DTLZ4):
        return evaluate_dtlz4(config, problem_state, X)
    if isinstance(config, DTLZ5):
        return evaluate_dtlz5(config, problem_state, X)
    if isinstance(config, DTLZ6):
        return evaluate_dtlz6(config, problem_state, X)
    if isinstance(config, DTLZ7):
        return evaluate_dtlz7(config, problem_state, X)
    raise TypeError(f"evaluate: unsupported DTLZ config type {type(config).__name__}")


def pf(config: DTLZConfig) -> etl.SymbolicTensor:
    """``pf(config) -> Pareto front tensor`` — routes to the DTLZ variant pf
    by config type (static ``isinstance``)."""
    if isinstance(config, DTLZ1):
        return pf_dtlz1(config)
    if isinstance(config, (DTLZ2, DTLZ3, DTLZ4)):
        return pf_dtlz2(config)
    if isinstance(config, DTLZ5):
        return pf_dtlz5(config)
    if isinstance(config, DTLZ6):
        return pf_dtlz6(config)
    if isinstance(config, DTLZ7):
        return pf_dtlz7(config)
    raise TypeError(f"pf: unsupported DTLZ config type {type(config).__name__}")
