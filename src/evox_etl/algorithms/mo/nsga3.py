"""Functional ETL port of the torch NSGA-III (``src/evox/algorithms/mo/nsga3.py``).

Plain functions only (no ``@etl.defn`` — DESIGN.md §4.3; ETL has no eager mode):
``init`` / ``init_ask`` / ``init_tell`` / ``ask`` / ``tell`` plus the frozen
``NSGA3Config`` / ``NSGA3State`` dataclasses. Call them only inside an active
trace (``etl.build`` / ``etl.evaluate``). The math is ported 1:1 from the torch
reference (read-only); the torch dynamic-shape boolean indexings in ``step`` are
replaced by fixed-shape mask tricks (``etl.select`` + one-hot ``reduce_max``
hits), and ``scatter_add`` becomes a one-hot sum.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from etl.core import SymbolicTensor

from evox_etl.algorithms._jit_fix_operator import _take_along_axis, clamp
from evox_etl.operators.crossover import simulated_binary
from evox_etl.operators.mutation import polynomial_mutation
from evox_etl.operators.sampling import uniform_sampling
from evox_etl.operators.selection import (
    non_dominate_rank,
    tournament_selection_multifit,
)


@dataclasses.dataclass(frozen=True)
class NSGA3Config:
    """NSGA-III hyperparameters (mirrors the torch ``__init__`` minus device).

    :param pop_size: The size of the population.
    :param n_objs: The number of objective functions.
    :param lb: The lower bounds for the decision variables (1D numpy array).
    :param ub: The upper bounds for the decision variables (1D numpy array).
    :param selection_op: The selection operation (optional; defaults to
        ``tournament_selection_multifit``).
    :param mutation_op: The mutation operation (optional; defaults to
        ``polynomial_mutation``).
    :param crossover_op: The crossover operation (optional; defaults to
        ``simulated_binary``).
    :param data_type: The data type for the decision variables (optional).
        Defaults to float32; ``bool`` draws the initial population as a
        uniform > 0.5 boolean tensor.
    """

    pop_size: int
    n_objs: int
    lb: np.ndarray
    ub: np.ndarray
    selection_op: Optional[Callable] = None
    mutation_op: Optional[Callable] = None
    crossover_op: Optional[Callable] = None
    data_type: Optional[Any] = None


def _config_flatten(config: NSGA3Config):
    """Zero-child flattening: the config travels as one opaque static node."""
    return [], config


def _config_unflatten(config: NSGA3Config, _children) -> NSGA3Config:
    return config


# ETL v1 rejects numpy arrays as static pytree leaves (they are neither
# TensorSpecs nor static Python values), so the config (which holds lb/ub as
# ndarrays) is registered as a childless pytree node carrying the whole
# config as its context — it then passes through etl.build/etl.run untouched.
etl.register_pytree_node(NSGA3Config, _config_flatten, _config_unflatten)


@dataclasses.dataclass(frozen=True)
class NSGA3State:
    """NSGA-III mutable state: tensor leaves only.

    ``off`` carries the offspring produced by the latest ``ask`` so that
    ``tell(config, state, fitness)`` (DESIGN.md §4.3) can merge the parents
    and offspring without receiving the candidates as an extra argument.
    """

    pop: SymbolicTensor  # (pop_size, dim) float32
    fit: SymbolicTensor  # (pop_size, n_objs) float32
    rank: SymbolicTensor  # (pop_size,) int32
    ref: SymbolicTensor  # (n_ref, n_objs) float32, Das-Dennis points
    off: SymbolicTensor  # (pop_size, dim) float32, latest offspring
    key: SymbolicTensor  # RNG key


def _masked_hit(n: int, idx: SymbolicTensor, mask: SymbolicTensor) -> SymbolicTensor:
    """One-hot position mask of the masked-in ``idx`` entries (shape (n,) bool).

    Compile-safe replacement for the torch ``_masked_assign`` rank update: a
    position ``p`` is hit iff some masked entry ``idx[j]`` (``mask[j]`` true)
    equals ``p``. Broadcasting over the full (n,) range keeps the output shape
    static (no dynamic-shape boolean indexing).
    """
    chosen = enp.logical_and(
        etl.equal(
            enp.expand_dims(enp.arange(n, dtype="int32"), 0),
            enp.expand_dims(idx, 1),
        ),
        enp.expand_dims(mask, 1),
    )
    return etl.ops.reduce_max(chosen, axes=0)


def init(config: NSGA3Config, key: SymbolicTensor) -> NSGA3State:
    """Draw the initial population (uniform within [lb, ub], or bool > 0.5)."""
    dim = config.lb.shape[0]
    lb = etl.ops.constant(etl.core.tensor(np.asarray(config.lb, dtype=np.float32)))
    ub = etl.ops.constant(etl.core.tensor(np.asarray(config.ub, dtype=np.float32)))
    k_pop, key = random.split(key)
    if config.data_type is bool:
        pop = random.uniform(k_pop, (config.pop_size, dim), 0.0, 1.0, "float32") > 0.5
    else:
        pop = random.uniform(k_pop, (config.pop_size, dim), 0.0, 1.0, "float32")
        pop = (ub - lb) * pop + lb
    fit = enp.full((config.pop_size, config.n_objs), float("inf"), dtype="float32")
    rank = enp.full((config.pop_size,), np.iinfo(np.int32).max, dtype="int32")
    off = enp.zeros(pop.shape, dtype=pop.dtype)
    ref, _ = uniform_sampling(config.pop_size, config.n_objs)
    return NSGA3State(pop=pop, fit=fit, rank=rank, ref=ref, off=off, key=key)


def init_ask(config: NSGA3Config, state: NSGA3State):
    """Return the full initial population for evaluation (no RNG)."""
    return state.pop, state


def init_tell(
    config: NSGA3Config, state: NSGA3State, fitness: SymbolicTensor
) -> NSGA3State:
    """Record the initial fitness and non-domination rank."""
    rank = non_dominate_rank(fitness)
    return dataclasses.replace(state, fit=fitness, rank=rank)


def ask(config: NSGA3Config, state: NSGA3State):
    """Produce the offspring batch: tournament selection + SBX + PM + clamp."""
    if config.selection_op is None:
        selection = tournament_selection_multifit
    else:
        selection = config.selection_op
    if config.crossover_op is None:
        crossover = simulated_binary
    else:
        crossover = config.crossover_op
    if config.mutation_op is None:
        mutation = polynomial_mutation
    else:
        mutation = config.mutation_op

    key, k_sel, k_cross, k_mut = random.split_n(state.key, 4)
    mating_pool = selection(k_sel, config.pop_size, [etl.cast(state.rank, "float32")])
    crossovered = crossover(k_cross, etl.gather(state.pop, mating_pool, axis=0))

    lb = etl.ops.constant(etl.core.tensor(np.asarray(config.lb, dtype=np.float32)))
    ub = etl.ops.constant(etl.core.tensor(np.asarray(config.ub, dtype=np.float32)))
    offspring = mutation(k_mut, crossovered, lb, ub)
    offspring = clamp(offspring, lb, ub)
    return offspring, dataclasses.replace(state, off=offspring, key=key)


def tell(config: NSGA3Config, state: NSGA3State, fitness: SymbolicTensor) -> NSGA3State:
    """Environmental selection (torch ``step`` lines 164-267, ported 1:1).

    The torch version filters ``rank <= worst_rank`` with dynamic boolean
    indexing; here the full (n, ...) tensors are kept and excluded rows are
    masked via ``etl.select`` instead.
    """
    pop_size = config.pop_size
    n_objs = config.n_objs
    dim = config.lb.shape[0]
    n = 2 * pop_size

    key, k_shuf, k_ref = random.split_n(state.key, 3)
    merge_pop = etl.concatenate([state.pop, state.off], axis=0)
    merge_fit = etl.concatenate([state.fit, fitness], axis=0)
    shuffled_idx = random.permutation(k_shuf, n, dtype=etl.int32)
    merge_pop = etl.gather(merge_pop, shuffled_idx, axis=0)
    merge_fit = etl.gather(merge_fit, shuffled_idx, axis=0)

    rank = non_dominate_rank(merge_fit)
    worst_rank = etl.topk(rank, pop_size + 1, axis=0, largest=False)[0][-1]

    # Filtered-set trick: mask excluded rows instead of indexing them out.
    in_mask = rank <= worst_rank  # (n,)
    ideal = etl.min(
        etl.select(enp.expand_dims(in_mask, 1), merge_fit, float("inf")), axes=0
    )
    norm_fit = etl.select(enp.expand_dims(in_mask, 1), merge_fit - ideal, float("inf"))

    # Extreme points (torch vmap_get_extreme, tensorized).
    weight = etl.eye(n_objs, dtype="float32") + 1e-6
    t = enp.expand_dims(norm_fit, 0) / enp.expand_dims(weight, 1)  # (n_objs, n, n_objs)
    ex_idx = etl.argmin(etl.max(t, axes=2), axis=1)  # (n_objs,)
    extreme = etl.gather(norm_fit, ex_idx, axis=0)  # (n_objs, n_objs)

    # Hyperplane intercepts. numpy linalg.solve RAISES on singular matrices,
    # so the solve must live behind etl.cond and never run unconditionally.
    def _hyperplane(extreme, norm_fit, in_mask, n_objs):
        return 1.0 / etl.solve(extreme, enp.full((n_objs,), 1.0, dtype="float32"))

    def _fallback(extreme, norm_fit, in_mask, n_objs):
        return etl.max(
            etl.select(enp.expand_dims(in_mask, 1), norm_fit, float("-inf")), axes=0
        )

    intercepts = etl.cond(
        etl.matrix_rank(extreme) == n_objs,
        _hyperplane,
        _fallback,
        extreme,
        norm_fit,
        in_mask,
        n_objs,
    )
    norm_fit = norm_fit / enp.expand_dims(intercepts, 0)

    # Distances by cosine similarity (torch _compute_distances). Excluded
    # rows are replaced by zeros first: their distances are never used
    # (group_id is overridden below) and inf/inf would yield NaN + warnings.
    n_ref = state.ref.shape[0]
    ref_shuffled = etl.gather(
        state.ref, random.permutation(k_ref, n_ref, dtype=etl.int32), axis=0
    )
    finite_fit = etl.select(enp.expand_dims(in_mask, 1), norm_fit, 0.0)
    mag = etl.norm(finite_fit, axis=1, keepdims=True)
    mag = enp.maximum(mag, 1e-10)
    fit_norm = finite_fit / mag
    ref_norm = ref_shuffled / enp.maximum(
        etl.norm(ref_shuffled, axis=1, keepdims=True), 1e-10
    )
    cosine_sim = etl.dot(fit_norm, etl.transpose(ref_norm, axes=(1, 0)))
    angular = etl.sqrt(enp.maximum(1.0 - etl.power(cosine_sim, 2.0), 1e-10))
    distances = mag * angular

    # Associate each solution with its nearest reference point.
    group_dist = etl.min(distances, axes=1)
    group_id = etl.argmin(distances, axis=1)
    upper_bound = n + dim + n_objs + 1  # sentinel > any valid index
    group_id = etl.cast(
        etl.select(rank == worst_rank, group_id, upper_bound), etl.int32
    )

    # rho / rho_last (one-hot sum replaces the torch scatter_add).
    arange_ref = enp.arange(n_ref, dtype="int32")
    one_hot = etl.equal(enp.expand_dims(arange_ref, 0), enp.expand_dims(group_id, 1))
    rho = etl.cast(
        etl.sum(
            etl.cast(
                enp.logical_and(one_hot, enp.expand_dims(rank < worst_rank, 1)),
                etl.int32,
            ),
            axes=0,
        ),
        etl.int32,
    )
    rho_last = etl.cast(
        etl.sum(
            etl.cast(
                enp.logical_and(one_hot, enp.expand_dims(rank == worst_rank, 1)),
                etl.int32,
            ),
            axes=0,
        ),
        etl.int32,
    )
    selected_num = etl.cast(etl.sum(rho, axes=None), etl.int32)
    rho = etl.cast(etl.select(rho_last == 0, upper_bound, rho), etl.int32)

    # First selection stage (rho == 0).
    row_indices = enp.arange(n_ref, dtype="int32")
    selected_ref = rho == 0
    d = etl.select(
        enp.logical_and(
            etl.equal(enp.expand_dims(group_id, 1), enp.expand_dims(row_indices, 0)),
            enp.expand_dims(rank == worst_rank, 1),
        ),
        enp.expand_dims(group_dist, 1),
        float("inf"),
    )  # (n, n_ref)
    candi_idx = etl.cast(etl.argmin(d, axis=0), etl.int32)  # (n_ref,)
    hit = _masked_hit(n, candi_idx, selected_ref)
    rank = etl.cast(
        etl.select(hit, etl.cast(worst_rank, etl.int32) - 1, rank), etl.int32
    )
    rho_last = etl.cast(etl.select(selected_ref, rho_last - 1, rho_last), etl.int32)
    rho = etl.cast(etl.select(selected_ref, 1, rho), etl.int32)
    rho = etl.cast(etl.select(rho_last == 0, upper_bound, rho), etl.int32)
    selected_num = etl.cast(
        selected_num
        + etl.cast(etl.sum(etl.cast(selected_ref, etl.int32), axes=None), etl.int32),
        etl.int32,
    )

    # Second selection stage: per-ref candidate tables.
    group_id = etl.cast(etl.select(hit, upper_bound, group_id), etl.int32)
    sel = etl.equal(enp.expand_dims(row_indices, 1), enp.expand_dims(group_id, 0))
    true_idx = etl.cast(
        etl.select(sel, enp.expand_dims(enp.arange(n, dtype="int32"), 0), upper_bound),
        etl.int32,
    )  # (n_ref, n); non-members sort as upper_bound at the end
    ref_candidates = etl.sort(true_idx, axis=1, stable=False)
    ref_cand_idx = enp.zeros((n_ref,), dtype="int32")

    def _cond(carry):
        rank, selected_num, ref_cand_idx, rho_last, rho, selected_ref, candi_idx = carry
        return selected_num < pop_size

    def _body(carry):
        rank, selected_num, ref_cand_idx, rho_last, rho, selected_ref, candi_idx = carry
        rho_level = etl.min(rho, axes=None)
        selected_ref = rho == rho_level
        candi_idx = enp.reshape(
            _take_along_axis(
                ref_candidates, enp.reshape(ref_cand_idx, (n_ref, 1)), axis=1
            ),
            (n_ref,),
        )
        hit = _masked_hit(n, candi_idx, selected_ref)
        rank = etl.cast(
            etl.select(hit, etl.cast(worst_rank, etl.int32) - 1, rank), etl.int32
        )
        ref_cand_idx = etl.cast(
            etl.select(selected_ref, ref_cand_idx + 1, ref_cand_idx), etl.int32
        )
        rho_last = etl.cast(
            etl.select(selected_ref, rho_last - 1, rho_last), etl.int32
        )
        rho = etl.cast(etl.select(selected_ref, rho_level + 1, rho), etl.int32)
        rho = etl.cast(etl.select(rho_last == 0, upper_bound, rho), etl.int32)
        selected_num = etl.cast(
            selected_num
            + etl.cast(
                etl.sum(etl.cast(selected_ref, etl.int32), axes=None), etl.int32
            ),
            etl.int32,
        )
        return (
            rank,
            selected_num,
            ref_cand_idx,
            rho_last,
            rho,
            selected_ref,
            candi_idx,
        )

    rank, selected_num, _, _, _, selected_ref, candi_idx = etl.while_loop(
        _cond,
        _body,
        (rank, selected_num, ref_cand_idx, rho_last, rho, selected_ref, candi_idx),
    )

    # Truncate the last batch back to pop_size (torch: demote the first
    # `dif` entries of the ascending sorted candidate indices).
    dif = selected_num - pop_size
    candi_idx = etl.cast(etl.select(selected_ref, candi_idx, upper_bound), etl.int32)
    sorted_index = etl.sort(candi_idx, axis=0, stable=False)
    sel_pos = enp.arange(n_ref, dtype="int32") < dif
    hit = _masked_hit(n, sorted_index, sel_pos)
    rank = etl.cast(etl.select(hit, worst_rank, rank), etl.int32)

    # Final survivors: exactly pop_size rows have rank < worst_rank (same set
    # as torch's boolean indexing; static-shape argsort keeps the shapes fixed).
    surv_mask = rank < worst_rank
    order = etl.argsort(
        etl.cast(etl.select(surv_mask, rank, upper_bound), etl.int32),
        axis=0,
        stable=True,
    )[:pop_size]
    pop = etl.gather(merge_pop, order, axis=0)
    fit = etl.gather(merge_fit, order, axis=0)
    rank = etl.gather(rank, order, axis=0)
    return dataclasses.replace(state, pop=pop, fit=fit, rank=rank, key=key)
