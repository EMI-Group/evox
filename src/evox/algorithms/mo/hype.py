"""HypE, part of the 2.0 functional API (plain functions).

Plain ``init/init_step/step`` functions following the step protocol of
``evox.core.algorithm`` (no ``@evoxc.defn`` — EvoX Compiler has no eager mode, see
DESIGN.md §4.3). One fused ``step`` does
selection → crossover → mutation → clamp →
``fitness = evaluate(offspring)`` → merge → rank → hypervolume truncation.
Because ``evaluate`` is an opaque closure that only returns fitness, the
offspring batch is carried in the ``offspring`` state leaf between the two
phases of the same trace (replaced by the next generation).
"""

from dataclasses import dataclass, replace
from typing import Callable, Optional

import evoxc
import evoxc.numpy as enp
import evoxc.random as random
from evoxc import core

from evox.algorithms._config_utils import ArrayLike, bake_bounds, normalize_bounds
from evox.operators.crossover import simulated_binary
from evox.operators.jit_fix_operator import clamp, lexsort
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import non_dominate_rank, tournament_selection


@dataclass(frozen=True)
class HypEConfig:
    """Config for HypE, mirroring the torch ``HypE.__init__`` signature (minus device).

    ``lb``/``ub`` are the per-dimension boundary values, accepted from any
    array-like input (list, tuple, or numpy array) and stored as flat tuples
    of plain Python floats (build it via :func:`make_hype`; the config is
    registered as an opaque childless pytree node because of the optional
    callable op fields below).

    Signature compatibility: the torch ``HypE.__init__`` accepts optional selection,
    mutation, and crossover operators. ``mutation_op``/``crossover_op`` are
    honored here; ``None`` means the torch default (``polynomial_mutation`` /
    ``simulated_binary``). A custom ``crossover_op`` is called as
    ``crossover_op(key, x)`` and a custom ``mutation_op`` as
    ``mutation_op(key, x, lb, ub)`` (the evoxc operators take an explicit RNG key
    and baked (dim,) float32 bounds). ``selection_op`` is deliberately NOT
    exposed: the 1.x implementation accepts it but immediately overwrites
    ``self.selection`` with ``tournament_selection``, so it is inert there too
    and this port keeps the hard-coded ``tournament_selection``.
    """

    pop_size: int
    n_objs: int
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    n_sample: int = 10000
    mutation_op: Optional[Callable] = None
    crossover_op: Optional[Callable] = None


def _config_flatten(config: HypEConfig):
    """Zero-child flattening: the config travels as one opaque static node."""
    return [], config


def _config_unflatten(config: HypEConfig, _children) -> HypEConfig:
    return config


# The config carries optional callable op fields (mutation_op/crossover_op):
# functions are not static pytree values (evoxc raises TraceError at a non-None
# callable leaf), so the config is registered as a childless pytree node and
# travels as one opaque static node through evoxc.build/evoxc.run.
evoxc.register_pytree_node(HypEConfig, _config_flatten, _config_unflatten)


def make_hype(
    pop_size: int,
    n_objs: int,
    lb: ArrayLike,
    ub: ArrayLike,
    n_sample: int = 10000,
    mutation_op: Optional[Callable] = None,
    crossover_op: Optional[Callable] = None,
) -> HypEConfig:
    """Build a ``HypEConfig``, normalizing ``lb``/``ub`` to flat float tuples.

    ``None`` op fields mean the torch defaults (``polynomial_mutation`` /
    ``simulated_binary``); a non-None field must be callable. Raises ValueError
    when a bound is not 1-D, ``lb``/``ub`` shapes differ, or an op field is not
    None and not callable.
    """
    for name, op in (("mutation_op", mutation_op), ("crossover_op", crossover_op)):
        if op is not None and not callable(op):
            raise ValueError(f"{name} must be callable or None, got {op!r}")
    lb, ub = normalize_bounds(lb, ub)
    return HypEConfig(
        pop_size=pop_size,
        n_objs=n_objs,
        lb=lb,
        ub=ub,
        n_sample=n_sample,
        mutation_op=mutation_op,
        crossover_op=crossover_op,
    )


@dataclass(frozen=True)
class HypEState:
    """Frozen state of HypE; all leaves are ``evoxc`` tensors.

    ``offspring`` holds the offspring batch between the generation phase and
    the selection phase of the same fused ``step`` trace (the opaque
    ``evaluate`` closure only returns fitness).
    """

    pop: core.SymbolicTensor
    fit: core.SymbolicTensor
    ref: core.SymbolicTensor
    offspring: core.SymbolicTensor
    key: core.Tensor


def cal_hv(
    key: core.Tensor,
    fit: core.SymbolicTensor,
    ref: core.SymbolicTensor,
    pop_size,
    n_sample: int,
) -> core.SymbolicTensor:
    """Monte-Carlo hypervolume contribution per solution (torch ``cal_hv`` + key).

    ``pop_size`` is a Python int in the first call and a scalar tensor in the
    merged call; ``fit`` is (n, m), ``ref`` (m,). Returns the estimated
    contribution (n,).
    """
    n, m = fit.shape
    alpha_num = evoxc.cumprod(
        evoxc.concatenate(
            [
                enp.ones((1,), dtype="float32"),
                (pop_size - enp.arange(1, n, dtype="float32"))
                / (n - enp.arange(1, n, dtype="float32")),
            ],
            axis=0,
        ),
        axis=0,
    )
    alpha = evoxc.nan_to_num(alpha_num / enp.arange(1, n + 1, dtype="float32"))

    f_min = evoxc.min(fit, axes=0)

    samples = (
        random.uniform(key, (n_sample, m), 0.0, 1.0, "float32") * (ref - f_min)
        + f_min
    )

    # (n_sample, n, m) <= 0 reduced with all() over the last axis → (n_sample, n)
    pds = evoxc.ops.reduce_min(
        evoxc.less_equal(
            enp.expand_dims(fit, 0) - enp.expand_dims(samples, 1), 0.0
        ),
        axes=2,
    )
    ds = evoxc.cast(evoxc.sum(evoxc.cast(pds, evoxc.int64), axes=1), evoxc.int64)
    ds = evoxc.select(ds == 0, ds, ds - 1)

    # torch where(pds.T, ds.unsqueeze(0), -1): (n, n_sample) int64
    temp = evoxc.cast(
        evoxc.select(evoxc.transpose(pds, axes=(1, 0)), enp.expand_dims(ds, 0), -1),
        evoxc.int64,
    )
    # torch indexes alpha[temp] with -1 wrapping to the last entry; clamp the
    # index instead — the value is masked out below anyway.
    safe_idx = evoxc.maximum(temp, 0)
    gathered = evoxc.gather(alpha, safe_idx, axis=0)
    value = evoxc.select(evoxc.not_equal(temp, -1), gathered, 0.0)
    f = evoxc.sum(value, axes=1)

    return f * evoxc.prod(ref - f_min) / float(n_sample)


def init(config: HypEConfig, key: core.Tensor) -> HypEState:
    """Draw the initial population and return the initial state."""
    lb, ub = bake_bounds(config.lb, config.ub)
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


def init_step(config: HypEConfig, state: HypEState, evaluate):
    """Generation 0 (fused torch ``init_step``): evaluate the FULL population
    and derive ``ref = 1.2 * max(fitness)``."""
    fitness = evaluate(state.pop)
    ref = enp.full((config.n_objs,), 1.2, dtype="float32") * evoxc.max(
        fitness, axes=None
    )
    return replace(state, fit=fitness, ref=ref)


def step(config: HypEConfig, state: HypEState, evaluate):
    """Run ONE full HypE generation (fused torch ``step``).

    Phase 1 (offspring generation, torch ``step`` lines 125-130): hypervolume-
    contribution tournament selection → crossover (``simulated_binary`` unless
    ``config.crossover_op``) → mutation (``polynomial_mutation`` unless
    ``config.mutation_op``) → clamp, offspring stored in the intermediate
    state. Phase 2: ``fitness = evaluate(offspring)`` through the
    workflow-owned opaque closure. Phase 3 (selection, torch ``step`` lines
    132-146): merge parents and offspring, truncate by non-domination rank +
    hypervolume.
    """
    lb, ub = bake_bounds(config.lb, config.ub)

    key, k_hv, k_sel, k_cross, k_mut = random.split_n(state.key, 5)

    hv = cal_hv(k_hv, state.fit, state.ref, config.pop_size, config.n_sample)
    # torch selects on -hv: highest hypervolume contribution wins the tournament
    mating_pool = tournament_selection(k_sel, config.pop_size, -hv)
    parents = evoxc.gather(state.pop, mating_pool, axis=0)
    crossover_fn = (
        config.crossover_op if config.crossover_op is not None else simulated_binary
    )
    crossovered = crossover_fn(k_cross, parents)
    mutation_fn = (
        config.mutation_op if config.mutation_op is not None else polynomial_mutation
    )
    offspring = mutation_fn(k_mut, crossovered, lb, ub)
    offspring = clamp(offspring, lb, ub)
    state = replace(state, offspring=offspring, key=key)

    fitness = evaluate(offspring)

    merge_pop = evoxc.concatenate([state.pop, state.offspring], axis=0)
    merge_fit = evoxc.concatenate([state.fit, fitness], axis=0)

    rank = non_dominate_rank(merge_fit)
    order = evoxc.argsort(rank, axis=0)
    worst_rank = evoxc.gather(
        rank, enp.expand_dims(order[config.pop_size - 1], 0), axis=0
    )[0]
    mask = rank <= worst_rank

    key, k_hv2 = random.split(state.key)
    k_count = (
        evoxc.cast(evoxc.sum(evoxc.cast(mask, evoxc.int32)), evoxc.int32) - config.pop_size
    )
    hv = cal_hv(k_hv2, merge_fit, state.ref, k_count, config.n_sample)
    dis = evoxc.select(mask, hv, float("-inf"))

    # last key primary (rank), -dis as tiebreaker — torch 1:1
    combined_indices = lexsort([-dis, rank])[: config.pop_size]

    pop = evoxc.gather(merge_pop, combined_indices, axis=0)
    fit = evoxc.gather(merge_fit, combined_indices, axis=0)

    return replace(state, pop=pop, fit=fit, key=key)
