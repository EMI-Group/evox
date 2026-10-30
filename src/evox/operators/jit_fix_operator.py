"""Compatibility helpers (clamp/maximum/...) + shared gather util.

Plain (NOT ``@evoxc.defn``) functions.  Algorithms call these instead of raw
``evoxc`` ops wherever the classic code needed JIT-operator-fusion safety; the
math is identical (relu composition).

This module is their canonical home: ``evox.operators`` mirrors the classic
``operators/`` package surface, while these helpers (historically kept in
``evox/utils/``) live here.  All functions are plain Python: they may only be
called inside an active evoxc trace (the EvoX Compiler has no eager mode).

1.x name -> evoxc adaptation:
- ``switch`` — per-element select from a list of tensors; the torch
  ``torch.where`` chain becomes an ``evoxc.select`` chain (1:1).
- ``clamp``/``clamp_float``/``clamp_int`` — relu-based clamps (1:1).
- ``clip`` — clamp into [0, 1] (1:1; torch invokes ``clamp(a, 0, 1)``).
- ``maximum``/``minimum`` (+ ``_int``/``_float`` variants) — relu-based elementwise ops (1:1).
- ``lexsort(keys, dim=-1)`` — stable multi-key argsort; like the 1.x implementation,
  the LAST key is primary (numpy lexsort convention; verified against torch).
- ``nanmin``/``nanmax`` — NaN-ignoring min/max returning ``(values, indices)``.
- ``randint(key, low, high, size, dtype=None)`` — 1.x-style tensor-scalar
  randint; gains a leading ``key`` (evoxc RNG is stateless; the torch signature
  is ``randint(low, high, size, ...)``).
- ``_take_along_axis(x, indices, axis)`` — torch ``gather``/``take_along_axis``
  semantics for 1-D/2-D tensors (``evoxc.gather`` is numpy-take). Used by the
  algorithm ports for row/column-local selection. NOTE: an equivalent private
  helper lives in ``evox/operators/selection/non_dominate.py`` (pre-existing,
  kept separate to avoid touching the already-verified operators).
"""

from typing import Any, List, Optional, Sequence, Tuple, Union

import evoxc
import evoxc.numpy as enp
import evoxc.random as random

Tensor = evoxc.SymbolicTensor

__all__ = [
    "switch",
    "clamp",
    "clamp_float",
    "clamp_int",
    "clip",
    "lexsort",
    "maximum",
    "maximum_float",
    "maximum_int",
    "minimum",
    "minimum_float",
    "minimum_int",
    "nanmax",
    "nanmin",
    "randint",
    "_take_along_axis",
]


def _relu(x: Tensor) -> Tensor:
    """Dtype-safe relu: ``evoxc.relu`` promotes int tensors to float64, so use select for ints."""
    if x.dtype == evoxc.int32 or x.dtype == evoxc.int64:
        return evoxc.select(x > 0, x, 0)
    return evoxc.relu(x)


def switch(label: Tensor, values: List[Tensor]) -> Tensor:
    """Select from `values` per element of `label` (jit-fix for the torch.where chain)."""
    value = values[0]
    for i in range(1, len(values)):
        value = evoxc.select(label <= i - 1, value, values[i])
    return value


def clamp(a: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
    """Clamp `a` elementwise into [lb, ub] via relu composition (jit-fix for torch.clamp)."""
    return a + _relu(lb - a) - _relu(a - ub)


def clamp_float(a: Tensor, lb: float, ub: float) -> Tensor:
    """Clamp `a` elementwise into scalar float bounds [lb, ub] (jit-fix for torch.clamp)."""
    return a + evoxc.relu(lb - a) - evoxc.relu(a - ub)


def clamp_int(a: Tensor, lb: int, ub: int) -> Tensor:
    """Clamp `a` elementwise into scalar int bounds [lb, ub], preserving `a`'s dtype."""
    return evoxc.cast(a + _relu(lb - a) - _relu(a - ub), a.dtype)


def clip(a: Tensor) -> Tensor:
    """Clip `a` elementwise into [0, 1] (jit-fix for torch.clamp; invokes clamp(a, 0, 1))."""
    return clamp_float(a, 0.0, 1.0)


def maximum(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise maximum of `a` and `b` via relu composition (jit-fix for torch.maximum)."""
    return a + _relu(b - a)


def minimum(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise minimum of `a` and `b` via relu composition (jit-fix for torch.minimum)."""
    return a - _relu(a - b)


def maximum_int(a: Tensor, b: int) -> Tensor:
    """Elementwise maximum of `a` and scalar int `b`, preserving `a`'s dtype."""
    return evoxc.cast(a + _relu(b - a), a.dtype)


def minimum_int(a: Tensor, b: int) -> Tensor:
    """Elementwise minimum of `a` and scalar int `b`, preserving `a`'s dtype."""
    return evoxc.cast(a - _relu(a - b), a.dtype)


def maximum_float(a: Tensor, b: float) -> Tensor:
    """Elementwise maximum of `a` and scalar float `b` (jit-fix for torch.maximum)."""
    return a + evoxc.relu(b - a)


def minimum_float(a: Tensor, b: float) -> Tensor:
    """Elementwise minimum of `a` and scalar float `b` (jit-fix for torch.minimum)."""
    return a - evoxc.relu(a - b)


def _gather_last(x: Tensor, indices: Tensor) -> Tensor:
    """Per-position gather along the last axis (evoxc.gather has numpy-take semantics)."""
    shape = x.shape
    n = 1
    for s in shape:
        n *= s
    flat = enp.reshape(x, (-1,))
    base = enp.reshape(enp.arange(0, n, shape[-1]), tuple(shape[:-1]) + (1,))
    flat_idx = base + indices
    return evoxc.gather(flat, flat_idx, axis=0)


def _torch_gather(x: Tensor, indices: Tensor, dim: int) -> Tensor:
    """torch.gather(x, dim, indices) for same-rank operands, built on evoxc.gather."""
    rank = len(x.shape)
    axis = dim if dim >= 0 else dim + rank
    if axis == rank - 1:
        return _gather_last(x, indices)
    perm = tuple(i for i in range(rank) if i != axis) + (axis,)
    inv = tuple(sorted(range(rank), key=lambda i: perm[i]))
    x_t = evoxc.transpose(x, axes=perm)
    idx_t = evoxc.transpose(indices, axes=perm)
    return evoxc.transpose(_gather_last(x_t, idx_t), axes=inv)


def lexsort(keys: List[Tensor], dim: int = -1) -> Tensor:
    """Indices sorting along `dim` lexicographically (LAST key primary), stable.

    Stable-argsort by ``keys[0]``, then refine with each later key; the last key
    ends up the primary sort key (numpy lexsort convention); earlier keys only
    break ties.
    """
    axis = dim % len(keys[0].shape)
    sorted_indices = evoxc.argsort(keys[0], axis=axis, stable=True)
    for key in keys[1:]:
        sorted_key = _torch_gather(key, sorted_indices, axis)
        final_sorted_indices = evoxc.argsort(sorted_key, axis=axis, stable=True)
        sorted_indices = _torch_gather(sorted_indices, final_sorted_indices, axis)
    return sorted_indices


def nanmin(
    input_tensor: Tensor, dim: int = -1, keepdim: bool = False
) -> Tuple[Tensor, Tensor]:
    """Min along `dim` ignoring NaN; returns a (values, indices) tuple."""
    mask = evoxc.isnan(input_tensor)
    clean = evoxc.select(
        mask,
        enp.full(tuple(input_tensor.shape), float("inf"), dtype=input_tensor.dtype),
        input_tensor,
    )
    values = evoxc.min(clean, axes=dim, keepdims=keepdim)
    indices = evoxc.argmin(clean, axis=dim, keepdims=keepdim)
    return values, indices


def nanmax(
    input_tensor: Tensor, dim: int = -1, keepdim: bool = False
) -> Tuple[Tensor, Tensor]:
    """Max along `dim` ignoring NaN; returns a (values, indices) tuple."""
    mask = evoxc.isnan(input_tensor)
    clean = evoxc.select(
        mask,
        enp.full(tuple(input_tensor.shape), float("-inf"), dtype=input_tensor.dtype),
        input_tensor,
    )
    values = evoxc.max(clean, axes=dim, keepdims=keepdim)
    indices = evoxc.argmax(clean, axis=dim, keepdims=keepdim)
    return values, indices


def randint(
    key: Tensor,
    low: Union[int, Tensor],
    high: Union[int, Tensor],
    size: Sequence[int],
    dtype: Optional[Any] = None,
) -> Tensor:
    """Keyed randint like torch's tensor-scalar randint; deterministic for a fixed key.

    `size` must be a static tuple of ints. With Python-int bounds this calls
    `random.randint` directly (dtype=None yields evoxc's int32 default); with
    tensor-scalar bounds it scales `random.uniform` draws exactly like the torch
    reference (uniform float32, truncating cast).
    """
    if isinstance(low, int) and isinstance(high, int):
        return random.randint(key, size, low, high, dtype or evoxc.int32)
    if isinstance(low, int):
        dtype = dtype or high.dtype
        u = random.uniform(key, size, 0.0, 1.0, evoxc.float32)
        r = evoxc.cast(high - low, evoxc.float32)
        return evoxc.cast(low + evoxc.cast(u * r, dtype), dtype)
    if dtype is None:
        dtype = low.dtype
    u = random.uniform(key, size, 0.0, 1.0, evoxc.float32)
    r = evoxc.cast(high - low, evoxc.float32)
    return low + evoxc.cast(u * r, dtype)


def _take_along_axis(x: Tensor, indices: Tensor, axis: int) -> Tensor:
    """Take entries of ``x`` along ``axis`` at ``indices`` (torch ``take_along_axis``).

    Result shape equals ``indices.shape`` (``x`` must be 1-D or 2-D with a static
    shape); ``evoxc.gather`` has numpy-take semantics, so per-row/per-column 2-D
    gathers go through a row-major flatten. Equivalent to the private helper in
    ``evox/operators/selection/non_dominate.py``.
    """
    if len(x.shape) == 1:
        return evoxc.gather(x, evoxc.cast(indices, "int64"), axis=0)
    if axis == 0:
        m = x.shape[1]
        idx_flat = evoxc.cast(indices, "int64") * m + evoxc.reshape(
            enp.arange(m, dtype="int64"), (1, m)
        )
        return evoxc.gather(evoxc.reshape(x, (-1,)), idx_flat)
    if axis == 1:
        s = x.shape[1]
        idx_flat = evoxc.reshape(
            enp.arange(x.shape[0], dtype="int64"), (x.shape[0], 1)
        ) * s + evoxc.cast(indices, "int64")
        return evoxc.gather(evoxc.reshape(x, (-1,)), idx_flat)
    raise ValueError(f"_take_along_axis: axis {axis} not supported")
