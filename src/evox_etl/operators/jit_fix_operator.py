"""ETL port of the torch evox ``utils/jit_fix_operator.py`` helpers + shared gather util.

Functional (plain, NOT ``@etl.defn``) ports of the torch helper functions in
``src/evox/utils/jit_fix_operator.py`` (read-only reference). The torch
algorithms call these instead of raw ``torch.clamp``/``torch.maximum``/... for
JIT-operator-fusion safety; the etl ports keep the exact same math (relu
composition), which the algorithm ports rely on for torch parity.

Canonical evox_etl.operators only mirrors the torch ``operators/`` package, so
these utils (which torch keeps in ``evox/utils/``) had no canonical home — this
module is it. All functions are plain Python: they may only be called inside an
active etl trace (ETL has no eager mode).

Torch name -> etl adaptation:
- ``clamp``/``clamp_float``/``clamp_int`` — relu-based clamps (1:1).
- ``maximum``/``minimum`` (+ ``_int`` variants) — relu-based elementwise ops (1:1).
- ``lexsort(keys, dim=-1)`` — stable multi-key argsort; like the torch original,
  the LAST key is primary (numpy lexsort convention; verified against torch).
- ``nanmin``/``nanmax`` — NaN-ignoring min/max returning ``(values, indices)``.
- ``randint(key, low, high, size, dtype=None)`` — torch-style tensor-scalar
  randint; gains a leading ``key`` (etl RNG is stateless; the torch signature
  is ``randint(low, high, size, ...)``).
- ``_take_along_axis(x, indices, axis)`` — torch ``gather``/``take_along_axis``
  semantics for 1-D/2-D tensors (``etl.gather`` is numpy-take). Used by the
  algorithm ports for row/column-local selection. NOTE: an equivalent private
  helper lives in ``evox_etl/operators/selection/non_dominate.py`` (pre-existing,
  kept separate to avoid touching torch-parity-verified operators).
"""

from typing import Any, List, Optional, Sequence, Tuple, Union

import etl
import etl.numpy as enp
import etl.random as random

Tensor = etl.SymbolicTensor

__all__ = [
    "clamp",
    "clamp_float",
    "clamp_int",
    "lexsort",
    "maximum",
    "maximum_int",
    "minimum",
    "minimum_int",
    "nanmax",
    "nanmin",
    "randint",
    "_take_along_axis",
]


def _relu(x: Tensor) -> Tensor:
    """Dtype-safe relu: ``etl.relu`` promotes int tensors to float64, so use select for ints."""
    if x.dtype == etl.int32 or x.dtype == etl.int64:
        return etl.select(x > 0, x, 0)
    return etl.relu(x)


def clamp(a: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
    """Clamp `a` elementwise into [lb, ub] via relu composition (jit-fix for torch.clamp)."""
    return a + _relu(lb - a) - _relu(a - ub)


def clamp_float(a: Tensor, lb: float, ub: float) -> Tensor:
    """Clamp `a` elementwise into scalar float bounds [lb, ub] (jit-fix for torch.clamp)."""
    return a + etl.relu(lb - a) - etl.relu(a - ub)


def clamp_int(a: Tensor, lb: int, ub: int) -> Tensor:
    """Clamp `a` elementwise into scalar int bounds [lb, ub], preserving `a`'s dtype."""
    return etl.cast(a + _relu(lb - a) - _relu(a - ub), a.dtype)


def maximum(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise maximum of `a` and `b` via relu composition (jit-fix for torch.maximum)."""
    return a + _relu(b - a)


def minimum(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise minimum of `a` and `b` via relu composition (jit-fix for torch.minimum)."""
    return a - _relu(a - b)


def maximum_int(a: Tensor, b: int) -> Tensor:
    """Elementwise maximum of `a` and scalar int `b`, preserving `a`'s dtype."""
    return etl.cast(a + _relu(b - a), a.dtype)


def minimum_int(a: Tensor, b: int) -> Tensor:
    """Elementwise minimum of `a` and scalar int `b`, preserving `a`'s dtype."""
    return etl.cast(a - _relu(a - b), a.dtype)


def _gather_last(x: Tensor, indices: Tensor) -> Tensor:
    """Per-position gather along the last axis (etl.gather has numpy-take semantics)."""
    shape = x.shape
    n = 1
    for s in shape:
        n *= s
    flat = enp.reshape(x, (-1,))
    base = enp.reshape(enp.arange(0, n, shape[-1]), tuple(shape[:-1]) + (1,))
    flat_idx = base + indices
    return etl.gather(flat, flat_idx, axis=0)


def _torch_gather(x: Tensor, indices: Tensor, dim: int) -> Tensor:
    """torch.gather(x, dim, indices) for same-rank operands, built on etl.gather."""
    rank = len(x.shape)
    axis = dim if dim >= 0 else dim + rank
    if axis == rank - 1:
        return _gather_last(x, indices)
    perm = tuple(i for i in range(rank) if i != axis) + (axis,)
    inv = tuple(sorted(range(rank), key=lambda i: perm[i]))
    x_t = etl.transpose(x, axes=perm)
    idx_t = etl.transpose(indices, axes=perm)
    return etl.transpose(_gather_last(x_t, idx_t), axes=inv)


def lexsort(keys: List[Tensor], dim: int = -1) -> Tensor:
    """Indices sorting along `dim` lexicographically (LAST key primary), stable.

    Exact port of the torch evox ``lexsort`` (``src/evox/utils/jit_fix_operator.py``
    lines 216-252): stable-argsort by ``keys[0]``, then refine with each later key.
    Like the torch original, the last key ends up the primary sort key (numpy
    lexsort convention); earlier keys only break ties.
    """
    axis = dim % len(keys[0].shape)
    sorted_indices = etl.argsort(keys[0], axis=axis, stable=True)
    for key in keys[1:]:
        sorted_key = _torch_gather(key, sorted_indices, axis)
        final_sorted_indices = etl.argsort(sorted_key, axis=axis, stable=True)
        sorted_indices = _torch_gather(sorted_indices, final_sorted_indices, axis)
    return sorted_indices


def nanmin(
    input_tensor: Tensor, dim: int = -1, keepdim: bool = False
) -> Tuple[Tensor, Tensor]:
    """Min along `dim` ignoring NaN; returns a (values, indices) tuple."""
    mask = etl.isnan(input_tensor)
    clean = etl.select(
        mask,
        enp.full(tuple(input_tensor.shape), float("inf"), dtype=input_tensor.dtype),
        input_tensor,
    )
    values = etl.min(clean, axes=dim, keepdims=keepdim)
    indices = etl.argmin(clean, axis=dim, keepdims=keepdim)
    return values, indices


def nanmax(
    input_tensor: Tensor, dim: int = -1, keepdim: bool = False
) -> Tuple[Tensor, Tensor]:
    """Max along `dim` ignoring NaN; returns a (values, indices) tuple."""
    mask = etl.isnan(input_tensor)
    clean = etl.select(
        mask,
        enp.full(tuple(input_tensor.shape), float("-inf"), dtype=input_tensor.dtype),
        input_tensor,
    )
    values = etl.max(clean, axes=dim, keepdims=keepdim)
    indices = etl.argmax(clean, axis=dim, keepdims=keepdim)
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
    `random.randint` directly (dtype=None yields etl's int32 default); with
    tensor-scalar bounds it scales `random.uniform` draws exactly like the torch
    reference (uniform float32, truncating cast).
    """
    if isinstance(low, int) and isinstance(high, int):
        return random.randint(key, size, low, high, dtype or etl.int32)
    if isinstance(low, int):
        dtype = dtype or high.dtype
        u = random.uniform(key, size, 0.0, 1.0, etl.float32)
        r = etl.cast(high - low, etl.float32)
        return etl.cast(low + etl.cast(u * r, dtype), dtype)
    if dtype is None:
        dtype = low.dtype
    u = random.uniform(key, size, 0.0, 1.0, etl.float32)
    r = etl.cast(high - low, etl.float32)
    return low + etl.cast(u * r, dtype)


def _take_along_axis(x: Tensor, indices: Tensor, axis: int) -> Tensor:
    """Take entries of ``x`` along ``axis`` at ``indices`` (torch ``take_along_axis``).

    Result shape equals ``indices.shape`` (``x`` must be 1-D or 2-D with a static
    shape); ``etl.gather`` has numpy-take semantics, so per-row/per-column 2-D
    gathers go through a row-major flatten. Equivalent to the private helper in
    ``evox_etl/operators/selection/non_dominate.py``.
    """
    if len(x.shape) == 1:
        return etl.gather(x, etl.cast(indices, "int64"), axis=0)
    if axis == 0:
        m = x.shape[1]
        idx_flat = etl.cast(indices, "int64") * m + etl.reshape(
            enp.arange(m, dtype="int64"), (1, m)
        )
        return etl.gather(etl.reshape(x, (-1,)), idx_flat)
    if axis == 1:
        s = x.shape[1]
        idx_flat = etl.reshape(
            enp.arange(x.shape[0], dtype="int64"), (x.shape[0], 1)
        ) * s + etl.cast(indices, "int64")
        return etl.gather(etl.reshape(x, (-1,)), idx_flat)
    raise ValueError(f"_take_along_axis: axis {axis} not supported")
