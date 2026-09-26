"""Functional ETL port of the torch evox HypE algorithm (plain functions).

1:1 port of the read-only torch reference ``src/evox/algorithms/mo/hype.py``
as plain ``init/init_step/step`` functions following the step protocol of
``evox_etl.core.algorithm`` (no ``@etl.defn`` — ETL has no eager mode, see
DESIGN.md §4.3). The torch class does everything inside ``step()``; here the
fused ``step`` is selection → crossover → mutation → clamp →
``fitness = evaluate(offspring)`` → merge → rank → hypervolume truncation.
Because ``evaluate`` is an opaque closure that only returns fitness, the
offspring batch is carried in the ``offspring`` state leaf between the two
phases of the same trace (replaced by the next generation).
"""

from dataclasses import dataclass, replace

import etl
import etl.numpy as enp
import etl.random as random
from etl import core

from evox_etl.algorithms._config_utils import ArrayLike, bake_bounds, normalize_bounds
from evox_etl.operators.jit_fix_operator import clamp, lexsort
from evox_etl.operators.crossover import simulated_binary
from evox_etl.operators.mutation import polynomial_mutation
from evox_etl.operators.selection import non_dominate_rank, tournament_selection


@dataclass(frozen=True)
class HypEConfig:
    """Config for HypE, mirroring the torch ``HypE.__init__`` signature (minus device).

    ``lb``/``ub`` are the per-dimension boundary values, accepted from any
    array-like input (list, tuple, or numpy array) and stored as flat tuples
    of plain Python floats so the config stays a plain static-leaf pytree
    (build it via :func:`make_hype`).

    Signature parity: the torch ``HypE.__init__`` accepts optional selection,
    mutation, and crossover operators, but the evox_etl functional variant
    hard-codes the operators (non_dominate_rank, tournament_selection,
    simulated_binary, polynomial_mutation) — they are not config fields here.
    """

    pop_size: int
    n_objs: int
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    n_sample: int = 10000


def make_hype(
    pop_size: int,
    n_objs: int,
    lb: ArrayLike,
    ub: ArrayLike,
    n_sample: int = 10000,
) -> HypEConfig:
    """Build a ``HypEConfig``, normalizing ``lb``/``ub`` to flat float tuples.

    Raises ValueError when a bound is not 1-D or ``lb``/``ub`` shapes differ.
    """
    lb, ub = normalize_bounds(lb, ub)
    return HypEConfig(pop_size=pop_size, n_objs=n_objs, lb=lb, ub=ub, n_sample=n_sample)


@dataclass(frozen=True)
class HypEState:
    """Frozen state of HypE; all leaves are ETL tensors.

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
    ref = enp.full((config.n_objs,), 1.2, dtype="float32") * etl.max(
        fitness, axes=None
    )
    return replace(state, fit=fitness, ref=ref)


def step(config: HypEConfig, state: HypEState, evaluate):
    """Run ONE full HypE generation (fused torch ``step``).

    Phase 1 (old ``ask`` body, torch ``step`` lines 125-130): hypervolume-
    contribution tournament selection → SBX → polynomial mutation → clamp,
    offspring stored in the intermediate state. Phase 2:
    ``fitness = evaluate(offspring)`` through the workflow-owned opaque
    closure. Phase 3 (old ``tell`` body, torch ``step`` lines 132-146): merge
    parents and offspring, truncate by non-domination rank + hypervolume.
    """
    lb, ub = bake_bounds(config.lb, config.ub)

    key, k_hv, k_sel, k_cross, k_mut = random.split_n(state.key, 5)

    hv = cal_hv(k_hv, state.fit, state.ref, config.pop_size, config.n_sample)
    # torch selects on -hv: highest hypervolume contribution wins the tournament
    mating_pool = tournament_selection(k_sel, config.pop_size, -hv)
    parents = etl.gather(state.pop, mating_pool, axis=0)
    crossovered = simulated_binary(k_cross, parents)
    offspring = polynomial_mutation(k_mut, crossovered, lb, ub)
    offspring = clamp(offspring, lb, ub)
    state = replace(state, offspring=offspring, key=key)

    fitness = evaluate(offspring)

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
