"""Shared host-side config-normalization helpers for the evox_etl algorithm families.

Every algorithm family (`so/de_variants`, `so/pso_variants`, `so/es_variants`,
`mo`) duplicates the same small private config idioms: array-like fields
(`lb`, `ub`, `mean`, `stdev`, `center_init`, ...) are normalized to flat
tuples of plain Python floats (etl static pytree leaves must be Python values,
not numpy arrays), bounds pairs validated, scalar hyperparameters checked, and
the normalized values baked as float32 graph constants.  This module is the
canonical implementation; the per-module private helpers (`_to_float_tuple`,
`_bounds`, `_bake*`, inline ``tuple(float(v) for v in ...)`` bodies, bare
``assert``s) are redirected here by the family agents with no numeric change.

Semantics contract (1:1 with today's modules):
- `to_float_tuple` is dtype-PRESERVING by default (no float32 cast), matching
  the de/pso-family idioms: float32 numpy input stays float32-rounded because
  ``.tolist()`` yields the rounded Python floats, while float64/tuple input
  keeps full precision.  Pass ``dtype=np.float32`` to reproduce the es-family
  idiom ``tuple(np.asarray(x, dtype=np.float32).tolist())`` (cma_es, nes, and
  the conditional ndarray normalizers of ars/open_es/snes/des/guided_es/
  noise_reuse_es/persistent_es/esmc/asebo).
- Scalars (Python numbers and 0-d numpy arrays) are accepted as a 1-tuple —
  a deliberate superset (today's modules raise on scalar input) for the future
  ``make_*`` constructors.
- Validation raises ValueError with a clear message, not the bare ``assert``
  (AssertionError, dropped under ``python -O``) of today's modules.

Environment: numpy is used here on purpose — this module runs eagerly on the
host during config construction (``__post_init__``/``make_*``), outside any etl
trace, the one place numpy is allowed in the framework; the ``bake_*`` helpers
build etl graph constants from inside traced functions.  The module imports
``etl`` but no algorithm module, so importing it creates no import cycle.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import DTypeLike

import etl

ArrayLike = int | float | np.ndarray | Sequence[int | float]
"""Anything the array normalizers accept: a Python number, a numpy array, or a
sequence of numbers.  The implementation goes through ``np.asarray``, so numpy
scalars and any other np-convertible input also work."""


def to_float_tuple(value: ArrayLike, *, dtype: DTypeLike | None = None) -> tuple[float, ...]:
    """Normalize an array-like value to a flat tuple of plain Python floats.

    Consolidates: de.py ``_to_float_tuple``; the inline float-tuple bodies of
    jade/shade/sade/code.py; the ``np.asarray`` inline bodies of pso/cso/
    clpso/fs_pso/sl_pso_gs/sl_pso_us.py and the ``.ravel()`` variant of
    dms_pso_el.py; with ``dtype=np.float32``, the es-family
    ``tuple(np.asarray(x, dtype=np.float32).tolist())`` idiom (cma_es, nes,
    ars, open_es, snes, des, guided_es, noise_reuse_es, persistent_es, esmc,
    asebo).

    Dtype-preserving unless ``dtype`` is given.  Accepts scalars / 0-d numpy
    scalars as a 1-tuple (superset: today's modules raise on scalars) and
    flattens arrays of any shape (like dms_pso_el's ``ravel()``).  Values go
    through ``.tolist()`` + ``float()``, so float32 input keeps float32
    rounding and float64/tuple input keeps full precision.
    """
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return (float(arr),)
    return tuple(float(v) for v in arr.reshape(-1).tolist())


def normalize_bounds(lb: ArrayLike, ub: ArrayLike) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate a (lb, ub) pair and normalize both bounds to flat float tuples.

    Consolidates the bounds handling of every family: de.py/jade.py/shade.py/
    sade.py ``assert len(lb) == len(ub)`` plus normalization, and pso.py/
    cso.py/clpso.py/fs_pso.py/sl_pso_gs.py/sl_pso_us.py ``assert lb.ndim == 1
    and ub.ndim == 1 and lb.shape == ub.shape`` plus normalization.

    Raises ValueError (not the AssertionError of today's bare asserts) when a
    bound is not 1-D or the two shapes differ; returns the dtype-preserving
    flat tuples (see `to_float_tuple`).
    """
    lb_arr = np.asarray(lb)
    ub_arr = np.asarray(ub)
    if lb_arr.ndim != 1 or ub_arr.ndim != 1:
        raise ValueError(
            f"lb and ub must be 1-D, got shapes {lb_arr.shape} and {ub_arr.shape}"
        )
    if lb_arr.shape != ub_arr.shape:
        raise ValueError(
            f"lb and ub must have the same shape, got {lb_arr.shape} and {ub_arr.shape}"
        )
    return to_float_tuple(lb_arr), to_float_tuple(ub_arr)


def require_gt(name: str, value: int | float, bound: int | float) -> int | float:
    """Require ``value > bound``; raises ValueError, otherwise returns ``value``.

    Consolidates: cma_es.py ``assert self.sigma > 0``, open_es.py
    ``noise_stdev > 0`` / ``learning_rate > 0``, and the ``pop_size > 1`` /
    ``pop_size > 0`` asserts of the es family.
    """
    if not (value > bound):
        raise ValueError(f"{name} must be greater than {bound}, got {value!r}")
    return value


def require_ge(name: str, value: int | float, bound: int | float) -> int | float:
    """Require ``value >= bound``; raises ValueError, otherwise returns ``value``.

    Consolidates: de.py ``assert self.pop_size >= 4`` and shade.py/sade.py
    ``assert self.pop_size >= 9``.
    """
    if not (value >= bound):
        raise ValueError(f"{name} must be at least {bound}, got {value!r}")
    return value


def require_between(
    name: str,
    value: int | float,
    low: int | float,
    high: int | float,
    *,
    low_inclusive: bool = True,
    high_inclusive: bool = True,
) -> int | float:
    """Require ``low <= value <= high``; raises ValueError, else returns ``value``.

    Inclusivity of each end is controlled by ``low_inclusive`` /
    ``high_inclusive``.  Consolidates: de.py ``assert 0 < self.cross_probability
    <= 1`` (``low_inclusive=False``) and ars.py ``assert 0 <= self.elite_ratio
    <= 1`` (defaults).  A composite bound such as de.py's ``assert 1 <=
    self.num_difference_vectors < self.pop_size // 2`` is one call with
    ``low=1, high=pop_size // 2, high_inclusive=False``.
    """
    low_ok = value >= low if low_inclusive else value > low
    high_ok = value <= high if high_inclusive else value < high
    if not (low_ok and high_ok):
        left = "[" if low_inclusive else "("
        right = "]" if high_inclusive else ")"
        raise ValueError(f"{name} must be in {left}{low}, {high}{right}, got {value!r}")
    return value


def require_choice(name: str, value: Any, choices: Iterable[Any]) -> Any:
    """Require ``value in choices``; raises ValueError, otherwise returns ``value``.

    Consolidates: de.py ``assert self.base_vector in ["rand", "best"]``,
    open_es.py/esmc.py/asebo.py ``assert self.optimizer in (None, "adam")``
    and snes.py ``assert self.weight_type in ("recomb", "temp")``.
    """
    choices = tuple(choices)
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices!r}, got {value!r}")
    return value


def bake_float32_constant(
    values: ArrayLike, *, shape: Sequence[int] | None = None
) -> etl.SymbolicTensor:
    """Bake array-like values as a float32 graph constant.

    Consolidates the ``etl.ops.constant(etl.core.tensor(np.asarray(x, dtype=
    np.float32)))`` bodies of: de.py ``_constant_1d``, jade.py ``_bake_bounds``,
    sade.py ``_bake_lb_ub``, code.py ``_bake`` (default dtype), pso.py/cso.py/
    clpso.py/dms_pso_el.py ``_bounds``/``_bake`` (their ``[None, :]`` becomes
    ``shape=(1, -1)``), sl_pso_gs.py/sl_pso_us.py/fs_pso.py ``_bake_bounds``,
    nes.py ``_constant_1d`` and the mo/*.py ``_bounds`` helpers.

    Called from inside traced ``init``/``ask``/``tell`` functions exactly like
    today's module-local bakers.  ``shape`` applies a numpy ``reshape`` before
    ``etl.ops.constant``: ``(1, -1)`` reproduces the SO ``(1, dim)`` bound
    rows, ``None`` keeps the natural shape (code.py's (3, 2) ``param_pool``,
    the mo family's (dim,) bounds).
    """
    arr = np.asarray(values, dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(shape)
    return etl.ops.constant(etl.core.tensor(arr))


def bake_bounds(
    lb: ArrayLike, ub: ArrayLike, *, as_row: bool = False
) -> tuple[etl.SymbolicTensor, etl.SymbolicTensor]:
    """Bake a (lb, ub) pair as float32 graph constants of equal shape.

    Consolidates the bound-pair bakers of de.py/jade.py/sade.py/pso.py/
    cso.py/clpso.py/dms_pso_el.py/sl_pso_gs.py/sl_pso_us.py/fs_pso.py (pass
    ``as_row=True`` for their (1, dim) rows) and of nsga2.py/nsga3.py/moead.py/
    rvea.py/rveaa.py/hype.py (default: (dim,) constants).  shade.py's single-
    bound ``_bounds(bound, dim)`` is `bake_float32_constant` with
    ``shape=(1, -1)`` (its ``dim`` argument is unused today).
    """
    shape = (1, -1) if as_row else None
    return bake_float32_constant(lb, shape=shape), bake_float32_constant(ub, shape=shape)
