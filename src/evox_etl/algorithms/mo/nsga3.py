"""NSGA-III, part of the 2.0 functional API.

Plain functions only (no ``@evoxc.defn`` — DESIGN.md §4.3; EvoX Compiler has no eager mode):
``init`` / ``init_step`` / ``step`` plus the frozen ``NSGA3Config`` /
``NSGA3State`` dataclasses. ``step`` owns one whole generation (tournament
selection → SBX → polynomial mutation → ``evaluate`` → environmental
selection); call the functions only inside an active trace
(``evoxc.build`` / ``evoxc.evaluate``). The math matches the 1.x algorithm;
its dynamic-shape boolean indexings in the
environmental selection are replaced by fixed-shape mask tricks
(``evoxc.select`` + one-hot ``reduce_max`` hits), and ``scatter_add`` becomes
a one-hot sum.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

import numpy as np

import evoxc
import evoxc.numpy as enp
import evoxc.random as random
from evoxc.core import SymbolicTensor

from evox_etl.algorithms._config_utils import ArrayLike, bake_bounds, normalize_bounds
from evox_etl.operators.jit_fix_operator import _take_along_axis, clamp
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
    :param lb: The lower bounds for the decision variables (flat tuple of
        floats; accepted from any array-like via ``make_nsga3``).
    :param ub: The upper bounds for the decision variables (flat tuple of
        floats; accepted from any array-like via ``make_nsga3``).
    :param selection_op: The selection operation (optional; defaults to
        ``tournament_selection_multifit``).
    :param mutation_op: The mutation operation (optional; defaults to
        ``polynomial_mutation``).
    :param crossover_op: The crossover operation (optional; defaults to
        ``simulated_binary``).
    :param data_type: The data type for the decision variables: exactly ``None``
        (float32 population, the default) or the builtin ``bool`` type (the
        initial population is drawn as a uniform > 0.5 boolean tensor). Anything
        else — notably ``torch.bool``, unimportable in the evoxc port — raises
        ValueError (see ``_normalize_data_type``).
    """

    pop_size: int
    n_objs: int
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    selection_op: Optional[Callable] = None
    mutation_op: Optional[Callable] = None
    crossover_op: Optional[Callable] = None
    data_type: Optional[Any] = None


def _config_flatten(config: NSGA3Config):
    """Zero-child flattening: the config travels as one opaque static node."""
    return [], config


def _config_unflatten(config: NSGA3Config, _children) -> NSGA3Config:
    return config


# The config carries optional callable op fields (selection_op/mutation_op/
# crossover_op). Functions are not static pytree values (evoxc raises TraceError
# at a non-None callable leaf), so the config is registered as a childless
# pytree node and travels as one opaque static node through evoxc.build/evoxc.run.
# (ndarray leaves would now be accepted statically, but a non-None callable
# field anywhere in the pytree is not.)
evoxc.register_pytree_node(NSGA3Config, _config_flatten, _config_unflatten)


def _normalize_data_type(data_type: Optional[Any]) -> Optional[type]:
    """Validate ``data_type``: return ``bool`` or ``None`` (float32).

    Only ``None`` (float32 population, the default) and the builtin ``bool``
    type (bool population) are accepted; anything else — notably ``torch.bool``,
    which the evoxc port cannot import — raises ValueError. Callers wanting a
    boolean population must pass the builtin ``bool``.
    """
    if data_type is None or data_type is bool:
        return data_type
    raise ValueError(
        f"data_type must be None (float32) or the builtin bool, got "
        f"{data_type!r}. torch.bool is not supported in the evoxc port — pass "
        f"the builtin bool instead."
    )


def make_nsga3(
    pop_size: int,
    n_objs: int,
    lb: ArrayLike,
    ub: ArrayLike,
    selection_op: Optional[Callable] = None,
    mutation_op: Optional[Callable] = None,
    crossover_op: Optional[Callable] = None,
    data_type: Optional[Any] = None,
) -> NSGA3Config:
    """Construct an ``NSGA3Config`` from array-like bounds.

    ``lb``/``ub`` are accepted as any 1-D array-like and stored as flat float
    tuples; ``None`` op fields mean the algorithm default (see the config
    docstring). Raises ValueError when an op field is not callable, the
    bounds are not 1-D / shape-mismatched, or ``data_type`` is neither
    ``None`` nor the builtin ``bool``.
    """
    for name, op in (
        ("selection_op", selection_op),
        ("mutation_op", mutation_op),
        ("crossover_op", crossover_op),
    ):
        if op is not None and not callable(op):
            raise ValueError(f"{name} must be callable or None, got {op!r}")
    data_type = _normalize_data_type(data_type)
    lb, ub = normalize_bounds(lb, ub)
    return NSGA3Config(
        pop_size=pop_size,
        n_objs=n_objs,
        lb=lb,
        ub=ub,
        selection_op=selection_op,
        mutation_op=mutation_op,
        crossover_op=crossover_op,
        data_type=data_type,
    )


@dataclasses.dataclass(frozen=True)
class NSGA3State:
    """NSGA-III mutable state: tensor leaves only.

    ``off`` carries the offspring produced by the generation stage of
    ``step`` into its environmental-selection stage (both share one call;
    ``evaluate`` passes only fitness back, so the
    candidates travel through the intermediate state).
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
        evoxc.equal(
            enp.expand_dims(enp.arange(n, dtype="int32"), 0),
            enp.expand_dims(idx, 1),
        ),
        enp.expand_dims(mask, 1),
    )
    return evoxc.ops.reduce_max(chosen, axes=0)


def init(config: NSGA3Config, key: SymbolicTensor) -> NSGA3State:
    """Draw the initial population (uniform within [lb, ub], or bool > 0.5).

    ``config.data_type`` is re-validated so direct dataclass construction
    (bypassing ``make_nsga3``) fails loudly instead of degrading to floats.
    """
    data_type = _normalize_data_type(config.data_type)
    dim = len(config.lb)
    lb, ub = bake_bounds(config.lb, config.ub)
    k_pop, key = random.split(key)
    if data_type is bool:
        pop = random.uniform(k_pop, (config.pop_size, dim), 0.0, 1.0, "float32") > 0.5
    else:
        pop = random.uniform(k_pop, (config.pop_size, dim), 0.0, 1.0, "float32")
        pop = (ub - lb) * pop + lb
    fit = enp.full((config.pop_size, config.n_objs), float("inf"), dtype="float32")
    rank = enp.full((config.pop_size,), np.iinfo(np.int32).max, dtype="int32")
    off = enp.zeros(pop.shape, dtype=pop.dtype)
    ref, _ = uniform_sampling(config.pop_size, config.n_objs)
    return NSGA3State(pop=pop, fit=fit, rank=rank, ref=ref, off=off, key=key)


def init_step(
    config: NSGA3Config, state: NSGA3State, evaluate: Callable
) -> NSGA3State:
    """First generation: evaluate the FULL initial population (no RNG) and
    record its fitness and non-domination rank."""
    fitness = evaluate(state.pop)
    rank = non_dominate_rank(fitness)
    return dataclasses.replace(state, fit=fitness, rank=rank)


def step(config: NSGA3Config, state: NSGA3State, evaluate: Callable) -> NSGA3State:
    """Run ONE full generation: tournament selection + SBX + PM + clamp,
    ``evaluate`` the offspring batch, then environmental selection
    (torch ``step`` lines 157-267, ported 1:1).

    The torch version filters ``rank <= worst_rank`` with dynamic boolean
    indexing; here the full (n, ...) tensors are kept and excluded rows are
    masked via ``evoxc.select`` instead.
    """
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
    mating_pool = selection(k_sel, config.pop_size, [evoxc.cast(state.rank, "float32")])
    crossovered = crossover(k_cross, evoxc.gather(state.pop, mating_pool, axis=0))

    lb, ub = bake_bounds(config.lb, config.ub)
    offspring = mutation(k_mut, crossovered, lb, ub)
    offspring = clamp(offspring, lb, ub)
    intermediate = dataclasses.replace(state, off=offspring, key=key)

    fitness = evaluate(offspring)
    return _select(config, intermediate, fitness)


def _final_survivors(
    merge_pop: SymbolicTensor, merge_fit: SymbolicTensor, rank: SymbolicTensor, worst_rank: int, pop_size: int
) -> tuple[SymbolicTensor, SymbolicTensor, SymbolicTensor]:
    """Keep the ``pop_size`` rows with ``rank < worst_rank`` (final stage).

    Survivors are ordered by POSITION in the shuffled merge arrays, matching
    torch's boolean-mask selection ``merge_pop[rank < worst_rank]`` (the old
    behavior sorted them by ascending rank). The truncation stage guarantees
    exactly ``pop_size`` rows satisfy the mask, so a stable positional argsort
    with the rest pushed to a sentinel keeps the shapes static.
    """
    n = rank.shape[0]
    surv_mask = rank < worst_rank
    order = evoxc.argsort(
        evoxc.cast(evoxc.select(surv_mask, enp.arange(n, dtype="int32"), n + 1), evoxc.int32),
        axis=0,
        stable=True,
    )[:pop_size]
    pop = evoxc.gather(merge_pop, order, axis=0)
    fit = evoxc.gather(merge_fit, order, axis=0)
    rank = evoxc.gather(rank, order, axis=0)
    return pop, fit, rank


def _select(config: NSGA3Config, state: NSGA3State, fitness: SymbolicTensor) -> NSGA3State:
    """Environmental selection (torch ``step`` lines 164-267, ported 1:1).

    The torch version filters ``rank <= worst_rank`` with dynamic boolean
    indexing; here the full (n, ...) tensors are kept and excluded rows are
    masked via ``evoxc.select`` instead.
    """
    pop_size = config.pop_size
    n_objs = config.n_objs
    dim = len(config.lb)
    n = 2 * pop_size

    key, k_shuf, k_ref = random.split_n(state.key, 3)
    merge_pop = evoxc.concatenate([state.pop, state.off], axis=0)
    merge_fit = evoxc.concatenate([state.fit, fitness], axis=0)
    shuffled_idx = random.permutation(k_shuf, n, dtype=evoxc.int32)
    merge_pop = evoxc.gather(merge_pop, shuffled_idx, axis=0)
    merge_fit = evoxc.gather(merge_fit, shuffled_idx, axis=0)

    rank = non_dominate_rank(merge_fit)
    worst_rank = evoxc.topk(rank, pop_size + 1, axis=0, largest=False)[0][-1]

    # Filtered-set trick: mask excluded rows instead of indexing them out.
    in_mask = rank <= worst_rank  # (n,)
    ideal = evoxc.min(
        evoxc.select(enp.expand_dims(in_mask, 1), merge_fit, float("inf")), axes=0
    )
    norm_fit = evoxc.select(enp.expand_dims(in_mask, 1), merge_fit - ideal, float("inf"))

    # Extreme points (torch vmap_get_extreme, tensorized).
    weight = evoxc.eye(n_objs, dtype="float32") + 1e-6
    t = enp.expand_dims(norm_fit, 0) / enp.expand_dims(weight, 1)  # (n_objs, n, n_objs)
    ex_idx = evoxc.argmin(evoxc.max(t, axes=2), axis=1)  # (n_objs,)
    extreme = evoxc.gather(norm_fit, ex_idx, axis=0)  # (n_objs, n_objs)

    # Full-rank test (torch ``torch.linalg.matrix_rank(extreme) == n_objs``).
    # ``evoxc.matrix_rank``/``evoxc.svd`` are deferred by the stablehlo-v1
    # exporter (BackendError), so the rank is recomputed from the exportable
    # ``eigh``: the singular values of ``extreme`` are sqrt(eigvalsh(AᵀA)),
    # and rank = count(s > tol) with numpy/torch's default cutoff
    # tol = s_max * n * eps32. AᵀA is accumulated in float64 so the eigh
    # noise floor (~1e-8 * s_max) sits far below the cutoff (~1e-7 * s_max):
    # the test then agrees with SVD-based matrix_rank except for a razor-thin
    # band around the cutoff itself.
    extreme64 = evoxc.cast(extreme, "float64")
    gram = evoxc.dot(evoxc.transpose(extreme64, axes=(1, 0)), extreme64)
    evals, _ = evoxc.eigh(gram)  # ascending eigenvalues, >= 0 (AᵀA is PSD)
    sing = evoxc.sqrt(enp.maximum(evals, 0.0))
    tol = evoxc.max(sing, axes=None) * n_objs * float(np.finfo(np.float32).eps)
    full_rank = evoxc.sum(evoxc.cast(sing > tol, evoxc.int32), axes=None) == n_objs

    # Hyperplane intercepts. The full-rank guard above keeps the solve safe
    # for deficient ``extreme`` matrices (numpy linalg.solve would RAISE on
    # them); inside the guarded branch the solve runs on a full-rank matrix.
    def _hyperplane(extreme, norm_fit, in_mask, n_objs):
        # torch: hyperplane = solve(extreme, ones); intercepts = 1/hyperplane.
        # ``evoxc.solve`` is deferred by the stablehlo exporter too, so the
        # system is solved via the normal equations (AᵀA)⁻¹Aᵀ·b with an
        # eigh-based inverse — float64 keeps it near the numpy LU solve's
        # accuracy for the well-conditioned matrices that pass the guard.
        a64 = evoxc.cast(extreme, "float64")
        b64 = enp.full((n_objs, 1), 1.0, dtype="float64")
        gram = evoxc.dot(evoxc.transpose(a64, axes=(1, 0)), a64)
        evals, evecs = evoxc.eigh(gram)
        c = evoxc.dot(evoxc.transpose(a64, axes=(1, 0)), b64)  # (n_objs, 1)
        y = evoxc.dot(evoxc.transpose(evecs, axes=(1, 0)), c) / enp.reshape(
            evals, (n_objs, 1)
        )
        hyperplane = enp.reshape(evoxc.dot(evecs, y), (n_objs,))
        return evoxc.cast(1.0 / hyperplane, "float32")

    def _fallback(extreme, norm_fit, in_mask, n_objs):
        return evoxc.max(
            evoxc.select(enp.expand_dims(in_mask, 1), norm_fit, float("-inf")), axes=0
        )

    intercepts = evoxc.cond(
        full_rank,
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
    ref_shuffled = evoxc.gather(
        state.ref, random.permutation(k_ref, n_ref, dtype=evoxc.int32), axis=0
    )
    finite_fit = evoxc.select(enp.expand_dims(in_mask, 1), norm_fit, 0.0)
    mag = evoxc.norm(finite_fit, axis=1, keepdims=True)
    mag = enp.maximum(mag, 1e-10)
    fit_norm = finite_fit / mag
    ref_norm = ref_shuffled / enp.maximum(
        evoxc.norm(ref_shuffled, axis=1, keepdims=True), 1e-10
    )
    cosine_sim = evoxc.dot(fit_norm, evoxc.transpose(ref_norm, axes=(1, 0)))
    angular = evoxc.sqrt(enp.maximum(1.0 - evoxc.power(cosine_sim, 2.0), 1e-10))
    distances = mag * angular

    # Associate each solution with its nearest reference point.
    group_dist = evoxc.min(distances, axes=1)
    group_id = evoxc.argmin(distances, axis=1)
    upper_bound = n + dim + n_objs + 1  # sentinel > any valid index
    group_id = evoxc.cast(
        evoxc.select(rank == worst_rank, group_id, upper_bound), evoxc.int32
    )

    # rho / rho_last (one-hot sum replaces the torch scatter_add).
    arange_ref = enp.arange(n_ref, dtype="int32")
    one_hot = evoxc.equal(enp.expand_dims(arange_ref, 0), enp.expand_dims(group_id, 1))
    rho = evoxc.cast(
        evoxc.sum(
            evoxc.cast(
                enp.logical_and(one_hot, enp.expand_dims(rank < worst_rank, 1)),
                evoxc.int32,
            ),
            axes=0,
        ),
        evoxc.int32,
    )
    rho_last = evoxc.cast(
        evoxc.sum(
            evoxc.cast(
                enp.logical_and(one_hot, enp.expand_dims(rank == worst_rank, 1)),
                evoxc.int32,
            ),
            axes=0,
        ),
        evoxc.int32,
    )
    selected_num = evoxc.cast(evoxc.sum(rho, axes=None), evoxc.int32)
    rho = evoxc.cast(evoxc.select(rho_last == 0, upper_bound, rho), evoxc.int32)

    # First selection stage (rho == 0).
    row_indices = enp.arange(n_ref, dtype="int32")
    selected_ref = rho == 0
    d = evoxc.select(
        enp.logical_and(
            evoxc.equal(enp.expand_dims(group_id, 1), enp.expand_dims(row_indices, 0)),
            enp.expand_dims(rank == worst_rank, 1),
        ),
        enp.expand_dims(group_dist, 1),
        float("inf"),
    )  # (n, n_ref)
    candi_idx = evoxc.cast(evoxc.argmin(d, axis=0), evoxc.int32)  # (n_ref,)
    hit = _masked_hit(n, candi_idx, selected_ref)
    rank = evoxc.cast(
        evoxc.select(hit, evoxc.cast(worst_rank, evoxc.int32) - 1, rank), evoxc.int32
    )
    rho_last = evoxc.cast(evoxc.select(selected_ref, rho_last - 1, rho_last), evoxc.int32)
    rho = evoxc.cast(evoxc.select(selected_ref, 1, rho), evoxc.int32)
    rho = evoxc.cast(evoxc.select(rho_last == 0, upper_bound, rho), evoxc.int32)
    selected_num = evoxc.cast(
        selected_num
        + evoxc.cast(evoxc.sum(evoxc.cast(selected_ref, evoxc.int32), axes=None), evoxc.int32),
        evoxc.int32,
    )

    # Second selection stage: per-ref candidate tables.
    group_id = evoxc.cast(evoxc.select(hit, upper_bound, group_id), evoxc.int32)
    sel = evoxc.equal(enp.expand_dims(row_indices, 1), enp.expand_dims(group_id, 0))
    true_idx = evoxc.cast(
        evoxc.select(sel, enp.expand_dims(enp.arange(n, dtype="int32"), 0), upper_bound),
        evoxc.int32,
    )  # (n_ref, n); non-members sort as upper_bound at the end
    ref_candidates = evoxc.sort(true_idx, axis=1, stable=False)
    ref_cand_idx = enp.zeros((n_ref,), dtype="int32")

    def _cond(carry):
        rank, selected_num, ref_cand_idx, rho_last, rho, selected_ref, candi_idx = carry
        return selected_num < pop_size

    def _body(carry):
        rank, selected_num, ref_cand_idx, rho_last, rho, selected_ref, candi_idx = carry
        rho_level = evoxc.min(rho, axes=None)
        selected_ref = rho == rho_level
        candi_idx = enp.reshape(
            _take_along_axis(
                ref_candidates, enp.reshape(ref_cand_idx, (n_ref, 1)), axis=1
            ),
            (n_ref,),
        )
        hit = _masked_hit(n, candi_idx, selected_ref)
        rank = evoxc.cast(
            evoxc.select(hit, evoxc.cast(worst_rank, evoxc.int32) - 1, rank), evoxc.int32
        )
        ref_cand_idx = evoxc.cast(
            evoxc.select(selected_ref, ref_cand_idx + 1, ref_cand_idx), evoxc.int32
        )
        rho_last = evoxc.cast(
            evoxc.select(selected_ref, rho_last - 1, rho_last), evoxc.int32
        )
        rho = evoxc.cast(evoxc.select(selected_ref, rho_level + 1, rho), evoxc.int32)
        rho = evoxc.cast(evoxc.select(rho_last == 0, upper_bound, rho), evoxc.int32)
        selected_num = evoxc.cast(
            selected_num
            + evoxc.cast(
                evoxc.sum(evoxc.cast(selected_ref, evoxc.int32), axes=None), evoxc.int32
            ),
            evoxc.int32,
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

    rank, selected_num, _, _, _, selected_ref, candi_idx = evoxc.while_loop(
        _cond,
        _body,
        (rank, selected_num, ref_cand_idx, rho_last, rho, selected_ref, candi_idx),
    )

    # Truncate the last batch back to pop_size (torch: demote the first
    # `dif` entries of the ascending sorted candidate indices).
    dif = selected_num - pop_size
    candi_idx = evoxc.cast(evoxc.select(selected_ref, candi_idx, upper_bound), evoxc.int32)
    sorted_index = evoxc.sort(candi_idx, axis=0, stable=False)
    sel_pos = enp.arange(n_ref, dtype="int32") < dif
    hit = _masked_hit(n, sorted_index, sel_pos)
    rank = evoxc.cast(evoxc.select(hit, worst_rank, rank), evoxc.int32)

    # Final survivors: exactly pop_size rows have rank < worst_rank (same set
    # as torch's boolean indexing). `_final_survivors` preserves the shuffled
    # merge order — torch `merge_pop[rank < worst_rank]` — instead of sorting
    # by ascending rank.
    pop, fit, rank = _final_survivors(merge_pop, merge_fit, rank, worst_rank, pop_size)
    return dataclasses.replace(state, pop=pop, fit=fit, rank=rank, key=key)
