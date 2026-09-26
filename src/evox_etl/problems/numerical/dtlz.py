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
- Sampling uses the canonical ``evox_etl.operators.sampling`` operators
  (``uniform_sampling`` / ``grid_sampling``) — they return a
  ``(tensor, n_samples)`` tuple, so call sites unpack ``[0]``.
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

from dataclasses import dataclass
from math import pi
from typing import Any

import etl
import etl.numpy as enp

from evox_etl.operators.sampling import grid_sampling, uniform_sampling


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


def _sample(config: DTLZConfig) -> etl.SymbolicTensor:
    """Base DTLZ sample — torch ``DTLZ.__init__``:
    ``uniform_sampling(ref_num * m, m)``."""
    return uniform_sampling(config.ref_num * config.m, config.m)[0]


def _reverse_axis1(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Axis-1 reversal without ``etl.flip`` (rejected by the compiled
    backends' StableHLO exporter): gather with descending static indices
    (``etl.gather`` has numpy ``take`` semantics). Requires a static ``k``
    (DTLZ shapes are fully static)."""
    k = x.shape[1]
    return etl.gather(x, enp.arange(k - 1, -1, -1, dtype=etl.int32), axis=1)


def _flip_cumprod_axis1(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """``flip(cumprod(concat([ones, x], axis=1)), 1)`` without ``etl.flip`` /
    ``etl.cumprod`` (both rejected by the compiled backends' StableHLO
    exporter). For a row ``[x0, ..., x_{k-1}]`` this yields the suffix
    products ``[prod(x0..x_{k-1}), ..., x0, 1]``: a prefix product scan
    (static Python loop — ``k`` is static) reversed along axis 1, with the
    trailing one appended. The prefix scan is sequential float multiply,
    so it is bit-identical to numpy/torch ``cumprod``.

    Requires a static ``k`` (DTLZ shapes are fully static)."""
    y = _reverse_axis1(_prefix_prod_axis1(x))
    ones = enp.ones((x.shape[0], 1), dtype=x.dtype)
    return enp.concatenate([y, ones], axis=1)


def _prefix_prod_axis1(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Prefix product along axis 1 via a static Python loop (``k`` static):
    column ``i`` holds ``x[:, 0] * ... * x[:, i]``."""
    acc = enp.ones((x.shape[0], 1), dtype=x.dtype)
    cols = []
    for i in range(x.shape[1]):
        acc = acc * x[:, i : i + 1]
        cols.append(acc)
    return enp.concatenate(cols, axis=1)


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
    flip_cumprod = _flip_cumprod_axis1(X[:, : m - 1])
    rest_part = enp.concatenate(
        [
            enp.ones((n, 1), dtype=X.dtype),
            1 - _reverse_axis1(X[:, : m - 1]),
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
        * _flip_cumprod_axis1(enp.maximum(enp.cos(X[:, : m - 1] * pi / 2), 0.0))
        * enp.concatenate(
            [
                enp.ones((X.shape[0], 1), dtype=X.dtype),
                enp.sin(_reverse_axis1(X[:, : m - 1]) * pi / 2),
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
        * _flip_cumprod_axis1(enp.maximum(enp.cos(X[:, : m - 1] * pi / 2), 0.0))
        * enp.concatenate(
            [
                enp.ones((n, 1), dtype=X.dtype),
                enp.sin(_reverse_axis1(X[:, : m - 1]) * pi / 2),
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
        * _flip_cumprod_axis1(enp.maximum(enp.cos(Xfront * pi / 2), 0.0))
        * enp.concatenate(
            [
                enp.ones((g.shape[0], 1), dtype=X.dtype),
                enp.sin(_reverse_axis1(Xfront) * pi / 2),
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
        * _flip_cumprod_axis1(enp.maximum(enp.cos(Xfront * pi / 2), 0.0))
        * enp.concatenate(
            [
                enp.ones((g.shape[0], 1), dtype=X.dtype),
                enp.sin(_reverse_axis1(Xfront) * pi / 2),
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
        * _flip_cumprod_axis1(enp.maximum(enp.cos(Xfront * pi / 2), 0.0))
        * enp.concatenate(
            [
                enp.ones((X.shape[0], 1), dtype=X.dtype),
                enp.sin(_reverse_axis1(Xfront) * pi / 2),
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

    x = grid_sampling(config.ref_num * config.m, m - 1)[0]

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
