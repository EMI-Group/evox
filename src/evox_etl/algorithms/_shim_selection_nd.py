"""Functional ETL port of the torch non-dominated sorting operators.

1:1 port of ``src/evox/operators/selection/non_dominate.py`` (read-only torch
reference) as plain functional ETL code. Functions are NOT decorated with
``@etl.defn`` — they are composed inside ``etl.trace``/``etl.build`` graphs
(see DESIGN.md §4.3). Numpy is used only for dtype/constant declarations.
"""
from __future__ import annotations

import numpy as np

import etl
import etl.numpy as enp

I32 = np.dtype("int32")
BOOL = np.dtype("bool")

__all__ = [
    "dominate_relation",
    "non_dominate_rank",
    "crowding_distance",
    "nd_environmental_selection",
]


def _take_axis(
    x: etl.SymbolicTensor, idx: etl.SymbolicTensor, axis: int
) -> etl.SymbolicTensor:
    """torch-style ``x.gather(axis, idx)`` for 1-D/2-D tensors of static shape.

    etl ``gather`` has numpy ``take`` semantics, so a per-column/per-row
    2-D gather is rewritten with a row-major flatten:
    ``out[i, j] = x[idx[i, j], j]`` (axis 0) or ``out[i, j] = x[i, idx[i, j]]``
    (axis 1).
    """
    if axis < 0:
        axis += len(x.shape)
    if len(x.shape) == 1:
        return etl.gather(x, idx, axis=axis)
    n, m = x.shape  # static shape required for the flatten trick
    if axis == 0:
        cols = enp.expand_dims(etl.arange(0, m, dtype=idx.dtype), 0)
        flat_idx = idx * m + cols
    else:
        rows = enp.expand_dims(etl.arange(0, n, dtype=idx.dtype), 1) * m
        flat_idx = rows + idx
    flat_x = etl.reshape(x, (n * m,))
    return etl.reshape(etl.gather(flat_x, flat_idx, axis=0), (n, m))


def _lexsort(keys: list[etl.SymbolicTensor], dim: int = -1) -> etl.SymbolicTensor:
    """Local lexsort (torch ``evox.utils.lexsort`` semantics: last key primary)."""
    idx = etl.argsort(keys[0], axis=dim, stable=True)
    for key in keys[1:]:
        sorted_key = _take_axis(key, idx, dim)
        final = etl.argsort(sorted_key, axis=dim, stable=True)
        idx = _take_axis(idx, final, dim)
    return idx


try:  # pragma: no cover - shared shim preferred once ported
    from ._shim_utils import lexsort
except ImportError:
    # _shim_utils not ported yet — local copy (torch evox.utils.lexsort).
    lexsort = _lexsort


def dominate_relation(
    x: etl.SymbolicTensor, y: etl.SymbolicTensor
) -> etl.SymbolicTensor:
    """Return the domination matrix A, where A_{ij} is True if x_i dominates y_j.

    :param x: Array with shape (n1, m) — one solution per row, m objectives.
    :param y: Array with shape (n2, m).
    :returns: Boolean domination relation matrix of shape (n1, n2).
    """
    x_expanded = enp.expand_dims(x, 1)  # (n1, 1, m)
    y_expanded = enp.expand_dims(y, 0)  # (1, n2, m)
    less_equal = etl.less_equal(x_expanded, y_expanded)
    strictly_less = etl.less(x_expanded, y_expanded)
    # .all(dim=2) -> reduce_min on bool, .any(dim=2) -> reduce_max on bool
    return etl.logical_and(
        etl.ops.reduce_min(less_equal, axes=2),
        etl.ops.reduce_max(strictly_less, axes=2),
    )


def update_dc_and_rank(
    dominate_relation_matrix: etl.SymbolicTensor,
    dominate_count: etl.SymbolicTensor,
    pareto_front: etl.SymbolicTensor,
    rank: etl.SymbolicTensor,
    current_rank: etl.SymbolicTensor,
):
    """Update ranks and domination counts for the current Pareto front.

    :returns: Updated ``(rank, dominate_count)``.
    """
    rank = etl.select(pareto_front, current_rank, rank)
    count_desc = etl.sum(
        etl.cast(enp.expand_dims(pareto_front, 1), I32)
        * etl.cast(dominate_relation_matrix, I32),
        axes=-2,
    )
    dominate_count = dominate_count - count_desc
    dominate_count = dominate_count - etl.cast(pareto_front, dominate_count.dtype)
    return rank, dominate_count


def non_dominate_rank(x: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Compute the non-domination rank (Pareto front index) of each solution.

    :param x: A 2D tensor where each row is a solution and each column an
        objective.
    :returns: A 1D int32 tensor with the non-domination rank of each solution.
    """
    n = x.shape[0]
    dominate_relation_matrix = dominate_relation(x, x)
    dominate_count = etl.sum(etl.cast(dominate_relation_matrix, I32), axes=0)
    rank = enp.zeros((n,), dtype=I32)
    pareto_front = etl.equal(dominate_count, 0)
    current_rank = enp.full((), np.int32(0), dtype=I32)

    def cond_fn(carry):
        rank, current_rank, dominate_count, pareto_front = carry
        return etl.ops.reduce_max(pareto_front, axes=None)  # pf.any()

    def body_fn(carry):
        rank, current_rank, dominate_count, pareto_front = carry
        rank, dominate_count = update_dc_and_rank(
            dominate_relation_matrix, dominate_count, pareto_front, rank, current_rank
        )
        current_rank = etl.cast(current_rank + 1, I32)
        new_pareto_front = etl.equal(dominate_count, 0)
        return rank, current_rank, dominate_count, new_pareto_front

    rank, _, _, _ = etl.while_loop(
        cond_fn, body_fn, (rank, current_rank, dominate_count, pareto_front)
    )
    return rank


def crowding_distance(
    costs: etl.SymbolicTensor, mask: etl.SymbolicTensor | None
) -> etl.SymbolicTensor:
    """Compute the crowding distance of each solution within its front.

    :param costs: A 2D tensor where each row is a solution and each column an
        objective.
    :param mask: A 1D boolean tensor selecting the solutions to consider
        (``None`` means all). Masked-out rows get ``-inf`` so they are
        selected last, matching the torch convention.
    :returns: A 1D tensor with the crowding distance of each solution.
    """
    n, m = costs.shape
    if mask is None:
        mask = enp.full((n,), True, dtype=BOOL)
    num_valid = etl.sum(etl.cast(mask, I32))  # int64 scalar
    inverted = etl.broadcast(
        enp.expand_dims(etl.cast(etl.logical_not(mask), costs.dtype), 1), (n, m)
    )
    rank = lexsort([costs, inverted], 0)
    costs = _take_axis(costs, rank, 0)
    distance_range = (
        etl.gather(costs, enp.expand_dims(num_valid - 1, 0), axis=0) - costs[0]
    )
    inner = (costs[2:] - costs[:-2]) / distance_range
    # distance[rank[0], :] = inf and distance[rank[num_valid - 1], :] = inf are
    # scatter replacements; etl scatter is put_along_axis (row assignment for a
    # 1-D index), so the per-column writes go through the inverse permutation
    # instead: dist[row p, j] = values[inv[p, j], j].
    values = enp.zeros((n, m), dtype=costs.dtype)
    values = etl.scatter(values, etl.arange(1, n - 1), inner, axis=0)
    inf_row = enp.full((m,), float("inf"), dtype=costs.dtype)
    values = etl.scatter(values, etl.arange(0, 1), enp.expand_dims(inf_row, 0), axis=0)
    inverse = etl.argsort(rank, axis=0)
    distance = _take_axis(values, inverse, 0)
    distance = etl.select(
        etl.equal(inverse, num_valid - 1), float("inf"), distance
    )
    crowding_distances = etl.select(
        enp.expand_dims(mask, 1), distance, float("-inf")
    )
    return etl.sum(crowding_distances, axes=1)


def nd_environmental_selection(
    x: etl.SymbolicTensor, f: etl.SymbolicTensor, topk: int
):
    """Select ``topk`` solutions by non-domination rank, then crowding distance.

    :returns: A tuple ``(x, f, rank, crowding_dis)`` of the selected solutions.
    """
    rank = non_dominate_rank(f)
    worst_rank = etl.max(etl.topk(rank, topk, axis=0, largest=False)[0], axes=None)
    mask = etl.equal(rank, worst_rank)
    crowding_dis = crowding_distance(f, mask)
    combined_order = lexsort([etl.negate(crowding_dis), rank])[:topk]
    return (
        etl.gather(x, combined_order, axis=0),
        etl.gather(f, combined_order, axis=0),
        etl.gather(rank, combined_order, axis=0),
        etl.gather(crowding_dis, combined_order, axis=0),
    )
