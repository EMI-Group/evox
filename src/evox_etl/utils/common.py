"""Functional ETL port of the JAX evox ``utils/common.py`` helpers (v0.9.0).

All functions here are PLAIN Python functions (NOT ``@etl.defn``) — callable
only inside an active etl trace, since ETL has no eager mode (a bare call
outside a trace raises ``etl.core.TraceError``, which is expected).  The JAX
``vmap`` batching becomes direct numpy-style broadcasting via ``expand_dims``:
an ``(n, d)`` × ``(k, d)`` pair becomes ``(n, 1, d)`` × ``(1, k, d)`` and the
per-pair function reduces over the last axis.  See DESIGN.md §4.3.
"""

from collections.abc import Iterable
from typing import Callable, List, Tuple, Union

import etl
import etl.numpy as enp
from etl import tree_flatten, tree_leaves, tree_map, tree_unflatten

Tensor = etl.SymbolicTensor

__all__ = [
    "min_by",
    "euclidean_dist",
    "manhattan_dist",
    "chebyshev_dist",
    "pairwise_func",
    "pairwise_euclidean_dist",
    "pairwise_manhattan_dist",
    "pairwise_chebyshev_dist",
    "pair_max",
    "cos_dist",
    "cal_max",
    "compose",
    "rank",
    "rank_based_fitness",
    "parse_opt_direction",
    "tree_flatten",
    "tree_leaves",
    "tree_map",
    "tree_unflatten",
]


def min_by(
    values: Union[Tensor, List[Tensor]],
    keys: Union[Tensor, List[Tensor]],
) -> Tuple[Tensor, Tensor]:
    """Find the value with the minimum key (lists are concatenated along axis 0).

    Returns the value row and the minimum key with the concatenation axis
    squeezed away (gather has numpy-take semantics, so a scalar index does
    that naturally).
    """
    if isinstance(values, list):
        values = etl.concatenate(values, axis=0)
        keys = etl.concatenate(keys, axis=0)
    min_index = etl.argmin(keys, axis=0)
    return etl.gather(values, min_index, axis=0), etl.gather(keys, min_index, axis=0)


def euclidean_dist(x: Tensor, y: Tensor) -> Tensor:
    """Euclidean distance between two vectors (reduces over the last axis)."""
    return etl.norm(x - y, axis=-1)


def manhattan_dist(x: Tensor, y: Tensor) -> Tensor:
    """Manhattan (L1) distance between two vectors (reduces over the last axis)."""
    return etl.sum(etl.abs(x - y), axes=-1)


def chebyshev_dist(x: Tensor, y: Tensor) -> Tensor:
    """Chebyshev (L-inf) distance between two vectors (reduces over the last axis)."""
    return etl.max(etl.abs(x - y), axes=-1)


def pairwise_func(
    x: Tensor,
    y: Tensor,
    func: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Apply ``func`` to every (x_i, y_j) pair via (n, 1, d) × (1, k, d) broadcast."""
    return func(enp.expand_dims(x, 1), enp.expand_dims(y, 0))


def pairwise_euclidean_dist(x: Tensor, y: Tensor) -> Tensor:
    """Pairwise euclidean distances: (n, d) × (k, d) -> (n, k)."""
    return pairwise_func(x, y, euclidean_dist)


def pairwise_manhattan_dist(x: Tensor, y: Tensor) -> Tensor:
    """Pairwise manhattan distances: (n, d) × (k, d) -> (n, k)."""
    return pairwise_func(x, y, manhattan_dist)


def pairwise_chebyshev_dist(x: Tensor, y: Tensor) -> Tensor:
    """Pairwise chebyshev distances: (n, d) × (k, d) -> (n, k)."""
    return pairwise_func(x, y, chebyshev_dist)


def pair_max(a: Tensor, b: Tensor) -> Tensor:
    """Max element of ``a - b`` (reduces over the last axis)."""
    return etl.max(a - b, axes=-1)


def cos_dist(x: Tensor, y: Tensor) -> Tensor:
    """Cosine similarity matrix between row vectors: (n, d) × (k, d) -> (n, k).

    Row-normalized dot products as in the JAX reference (zero-norm rows
    propagate nan like jnp — no silent clamping).
    """
    x_norm = x / etl.norm(x, axis=-1, keepdims=True)
    y_norm = y / etl.norm(y, axis=-1, keepdims=True)
    return etl.dot(x_norm, etl.transpose(y_norm, axes=(1, 0)))


def cal_max(x: Tensor, y: Tensor) -> Tensor:
    """Pairwise element max of differences: (n, d) × (k, d) -> (n, k)."""
    return pairwise_func(x, y, pair_max)


def compose(*functions):
    """Compose functions left-to-right; a single iterable argument is unwrapped."""
    if len(functions) == 1 and isinstance(functions[0], Iterable):
        functions = functions[0]

    def composed_function(carry):
        for function in functions:
            carry = function(carry)
        return carry

    return composed_function


def rank(array: Tensor) -> Tensor:
    """Rank (int) of each element of a 1-d array; ties broken by argsort order."""
    order = etl.argsort(array, axis=0)
    base = order * 0  # zero init in order's dtype (etl has no *_like ops)
    updates = enp.arange(array.shape[0], dtype=order.dtype)
    return etl.scatter(base, order, updates, axis=0)


def rank_based_fitness(raw_fitness: Tensor) -> Tensor:
    """Normalized rank in [-0.5, 0.5] from raw fitness (float32 output)."""
    num_elems = raw_fitness.shape[0]
    fitness_rank = etl.cast(rank(raw_fitness), etl.float32)
    return fitness_rank / (num_elems - 1) - 0.5


def parse_opt_direction(
    opt_direction: Union[str, Iterable[str]],
) -> Union[int, Tuple[int, ...]]:
    """Map ``"min"`` → 1 / ``"max"`` → -1; iterables become tuples of ±1.

    Pure Python — callable host-side and inside a trace.
    """
    if isinstance(opt_direction, str):
        if opt_direction == "min":
            return 1
        if opt_direction == "max":
            return -1
        raise ValueError(f"opt_direction is either 'min' or 'max', got {opt_direction}")
    if isinstance(opt_direction, Iterable):
        result = []
        for d in opt_direction:
            if d == "min":
                result.append(1)
            elif d == "max":
                result.append(-1)
            else:
                raise ValueError(f"opt_direction is either 'min' or 'max', got {d}")
        return tuple(result)
    raise ValueError(
        f"opt_direction should have type 'str' or 'Iterable[str]', got {type(opt_direction)}"
    )
