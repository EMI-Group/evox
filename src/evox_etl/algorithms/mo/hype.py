"""Functional ETL port of the torch evox HypE algorithm (plain functions).

1:1 port of the read-only torch reference ``src/evox/algorithms/mo/hype.py``
split into ``init/init_ask/init_tell/ask/tell`` plain functions (no
``@etl.defn`` — ETL has no eager mode, see DESIGN.md §4.3). The torch class
does everything inside ``step()``; here ``ask`` is the first half (selection
→ crossover → mutation → clamp) and ``tell`` the second half (merge → rank →
hypervolume truncation). Since the workflow contract passes only
``(config, state, fitness)`` to ``tell``, the offspring produced by ``ask``
is carried in the ``offspring`` state leaf (replaced by the next ``ask``).
"""

from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from etl import core

from evox_etl.algorithms._operator_shims import (
    clamp,
    lexsort,
    non_dominate_rank,
    polynomial_mutation,
    simulated_binary,
    tournament_selection,
)


@dataclass(frozen=True)
class HypEConfig:
    """Config for HypE, mirroring the torch ``HypE.__init__`` signature (minus device).

    ``lb``/``ub`` are 1-D boundary values; any ``np.asarray``-compatible
    sequence works. The config is registered below as a childless pytree
    node, so it travels through ``etl.build``/``etl.run`` as one opaque
    static node (etl v1's trace flattener would otherwise reject the
    ndarray leaves).
    """

    pop_size: int
    n_objs: int
    lb: np.ndarray
    ub: np.ndarray
    n_sample: int = 10000
    selection_op: Optional[Callable] = None
    mutation_op: Optional[Callable] = None
    crossover_op: Optional[Callable] = None


def _config_flatten(config: HypEConfig):
    """Zero-child flattening: the config travels as one opaque static node."""
    return [], config


def _config_unflatten(config: HypEConfig, _children) -> HypEConfig:
    return config


# ETL v1 rejects numpy arrays as static pytree leaves (they are neither
# TensorSpecs nor static Python values), so the config (which holds lb/ub as
# ndarrays) is registered as a childless pytree node carrying the whole
# config as its context — it then passes through etl.build/etl.run untouched.
etl.register_pytree_node(HypEConfig, _config_flatten, _config_unflatten)


@dataclass(frozen=True)
class HypEState:
    """Frozen state of HypE; all leaves are ETL tensors.

    ``offspring`` holds the ask-produced offspring between ``ask`` and
    ``tell`` (the workflow only forwards ``fitness`` to ``tell``).
    """

    pop: core.SymbolicTensor
    fit: core.SymbolicTensor
    ref: core.SymbolicTensor
    offspring: core.SymbolicTensor
    key: core.Tensor


def _bounds(config: HypEConfig):
    """Bake the boundary arrays as graph constants; returns ``(lb, ub)``."""
    lb = etl.ops.constant(core.tensor(np.asarray(config.lb, dtype=np.float32)))
    ub = etl.ops.constant(core.tensor(np.asarray(config.ub, dtype=np.float32)))
    return lb, ub


def cal_hv(
    key: core.Tensor,
    fit: core.SymbolicTensor,
    ref: core.SymbolicTensor,
    pop_size,
    n_sample: int,
) -> core.SymbolicTensor:
    """Monte-Carlo hypervolume contribution per solution (torch ``cal_hv`` + key).

    ``pop_size`` is a Python int (ask) or a scalar tensor (tell); ``fit`` is
    (n, m), ``ref`` (m,). Returns the estimated contribution (n,).
    """
    n, m = fit.shape
    alpha_num = etl.cumprod(
        etl.concatenate(
            [
                enp.ones((1,), dtype="float32"),
                (pop_size - enp.arange(1, n, dtype="float32"))
                / (n - enp.arange(1, n, dtype="float32")),
            ],
            axis=0,
        ),
        axis=0,
    )
    alpha = etl.nan_to_num(alpha_num / enp.arange(1, n + 1, dtype="float32"))

    f_min = etl.min(fit, axes=0)

    samples = (
        random.uniform(key, (n_sample, m), 0.0, 1.0, "float32") * (ref - f_min)
        + f_min
    )

    # (n_sample, n, m) <= 0 reduced with all() over the last axis → (n_sample, n)
    pds = etl.ops.reduce_min(
        etl.less_equal(
            enp.expand_dims(fit, 0) - enp.expand_dims(samples, 1), 0.0
        ),
        axes=2,
    )
    ds = etl.cast(etl.sum(etl.cast(pds, etl.int64), axes=1), etl.int64)
    ds = etl.select(ds == 0, ds, ds - 1)

    # torch where(pds.T, ds.unsqueeze(0), -1): (n, n_sample) int64
    temp = etl.cast(
        etl.select(etl.transpose(pds, axes=(1, 0)), enp.expand_dims(ds, 0), -1),
        etl.int64,
    )
    # torch indexes alpha[temp] with -1 wrapping to the last entry; clamp the
    # index instead — the value is masked out below anyway.
    safe_idx = etl.maximum(temp, 0)
    gathered = etl.gather(alpha, safe_idx, axis=0)
    value = etl.select(etl.not_equal(temp, -1), gathered, 0.0)
    f = etl.sum(value, axes=1)

    return f * etl.prod(ref - f_min) / float(n_sample)


def init(config: HypEConfig, key: core.Tensor) -> HypEState:
    """Draw the initial population and return the initial state."""
    lb, ub = _bounds(config)
    dim = len(config.lb)
    key, k_pop = random.split(key)
    population = (
        random.uniform(k_pop, (config.pop_size, dim), 0.0, 1.0, "float32")
        * (ub - lb)
        + lb
    )
    fit = enp.full((config.pop_size, config.n_objs), float("inf"), dtype="float32")
    ref = enp.ones((config.n_objs,), dtype="float32")
    offspring = enp.zeros((config.pop_size, dim), dtype="float32")
    return HypEState(
        pop=population, fit=fit, ref=ref, offspring=offspring, key=key
    )


def init_ask(config: HypEConfig, state: HypEState):
    """Return the FULL population for the generation-0 evaluation (no RNG)."""
    return state.pop, state


def init_tell(
    config: HypEConfig, state: HypEState, fitness: core.SymbolicTensor
) -> HypEState:
    """Record the initial fitness and derive ``ref = 1.2 * max(fitness)``."""
    fit = fitness
    ref = enp.full((config.n_objs,), 1.2, dtype="float32") * etl.max(
        fitness, axes=None
    )
    return replace(state, fit=fit, ref=ref)


def ask(config: HypEConfig, state: HypEState):
    """Produce the offspring batch (torch ``step`` lines 125-130)."""
    lb, ub = _bounds(config)
    boundary = enp.stack([lb, ub], axis=0)

    key, k_hv, k_sel, k_cross, k_mut = random.split_n(state.key, 5)

    hv = cal_hv(k_hv, state.fit, state.ref, config.pop_size, config.n_sample)
    # torch selects on -hv: highest hypervolume contribution wins the tournament
    mating_pool = tournament_selection(k_sel, config.pop_size, -hv)
    parents = etl.gather(state.pop, mating_pool, axis=0)
    crossovered = simulated_binary(k_cross, parents)
    offspring = polynomial_mutation(k_mut, crossovered, boundary)
    offspring = clamp(offspring, lb, ub)

    return offspring, replace(state, offspring=offspring, key=key)


def tell(
    config: HypEConfig, state: HypEState, fitness: core.SymbolicTensor
) -> HypEState:
    """Merge parents and offspring, truncate by non-domination rank + hypervolume
    (torch ``step`` lines 132-146)."""
    merge_pop = etl.concatenate([state.pop, state.offspring], axis=0)
    merge_fit = etl.concatenate([state.fit, fitness], axis=0)

    rank = non_dominate_rank(merge_fit)
    order = etl.argsort(rank, axis=0)
    worst_rank = etl.gather(
        rank, enp.expand_dims(order[config.pop_size - 1], 0), axis=0
    )[0]
    mask = rank <= worst_rank

    key, k_hv2 = random.split(state.key)
    k_count = (
        etl.cast(etl.sum(etl.cast(mask, etl.int32)), etl.int32) - config.pop_size
    )
    hv = cal_hv(k_hv2, merge_fit, state.ref, k_count, config.n_sample)
    dis = etl.select(mask, hv, float("-inf"))

    # last key primary (rank), -dis as tiebreaker — torch 1:1
    combined_indices = lexsort([-dis, rank])[: config.pop_size]

    pop = etl.gather(merge_pop, combined_indices, axis=0)
    fit = etl.gather(merge_fit, combined_indices, axis=0)

    return replace(state, pop=pop, fit=fit, key=key)
