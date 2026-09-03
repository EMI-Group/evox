"""Functional ETL ports of the torch evox `utils/jit_fix_operator.py` helpers.

These are plain (non-`@etl.defn`) functions: they may only be called inside an
active trace (e.g. a function passed to `etl.build`/`etl.evaluate`), since ETL
has no eager mode. Semantics mirror the torch originals exactly.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union

import etl
import etl.numpy as enp
import etl.random as random

Tensor = etl.SymbolicTensor


def _relu(x: Tensor) -> Tensor:
    """Dtype-safe relu: `etl.relu` promotes int tensors to float64, so use select for ints."""
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
    """Indices sorting rows along `dim` lexicographically (last key primary), stable."""
    sorted_indices = etl.argsort(keys[0], axis=dim, stable=True)
    for key in keys[1:]:
        sorted_key = _torch_gather(key, sorted_indices, dim)
        final_sorted_indices = etl.argsort(sorted_key, axis=dim, stable=True)
        sorted_indices = _torch_gather(sorted_indices, final_sorted_indices, dim)
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
