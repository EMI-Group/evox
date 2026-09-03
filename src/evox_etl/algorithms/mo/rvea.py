"""Functional ETL port of the torch evox RVEA algorithm (src/evox/algorithms/mo/rvea.py).

Reference Vector Guided Evolutionary Algorithm [1, 2] as plain `init/init_ask/
init_tell/ask/tell` functions over frozen `RVEAConfig`/`RVEAState` dataclasses
(DESIGN.md §4-5). No `@etl.defn` wrappers — call these only inside an active
trace (ETL has no eager mode). Ported 1:1 from the torch reference.

:references:
    [1] R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, "A reference vector guided
        evolutionary algorithm for many-objective optimization," IEEE TEVC, vol. 20,
        no. 5, pp. 773-791, 2016.
    [2] Z. Liang, T. Jiang, K. Sun, and R. Cheng, "GPU-accelerated Evolutionary
        Multiobjective Optimization Using Tensorized RVEA," GECCO '24, pp. 566-575.
"""

from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._operator_shims import (
    clamp,
    nanmax,
    nanmin,
    polynomial_mutation,
    randint,
    ref_vec_guided,
    simulated_binary,
    uniform_sampling,
)

Tensor = etl.SymbolicTensor


@dataclass(frozen=True)
class RVEAConfig:
    """Frozen RVEA hyperparameters (mirrors torch ``RVEA.__init__`` minus device)."""

    pop_size: int
    n_objs: int
    lb: np.ndarray
    ub: np.ndarray
    alpha: float = 2.0
    fr: float = 0.1
    max_gen: int = 100
    selection_op: Optional[Callable] = None
    mutation_op: Optional[Callable] = None
    crossover_op: Optional[Callable] = None


@dataclass(frozen=True)
class RVEAState:
    """Frozen RVEA state; every leaf is an ETL tensor (no Python scalars)."""

    pop: Tensor
    fit: Tensor
    reference_vector: Tensor
    init_v: Tensor
    gen: Tensor
    rv_adapt_every: Tensor
    key: Tensor


def init(config: RVEAConfig, key: Tensor) -> RVEAState:
    """Create the initial state: Das-Dennis reference vectors + uniform population."""
    v, n_v = uniform_sampling(config.pop_size, config.n_objs)
    dim = config.lb.shape[0]
    lb = etl.ops.constant(etl.core.tensor(np.asarray(config.lb, dtype=np.float32)))
    ub = etl.ops.constant(etl.core.tensor(np.asarray(config.ub, dtype=np.float32)))
    population = random.uniform(key, (n_v, dim), 0.0, 1.0, etl.float32) * (ub - lb) + lb
    fit = enp.full((n_v, config.n_objs), float("inf"))
    gen = enp.zeros((), dtype="int32")
    rv_adapt_every = enp.full(
        (), float(max(round(1.0 / config.fr), 1.0)), dtype="float32"
    )
    return RVEAState(
        pop=population,
        fit=fit,
        reference_vector=v,
        init_v=v,
        gen=gen,
        rv_adapt_every=rv_adapt_every,
        key=key,
    )


def init_ask(config: RVEAConfig, state: RVEAState) -> Tuple[Tensor, RVEAState]:
    """Return the full initial population (gen 0 evaluates the whole pop)."""
    return state.pop, state


def init_tell(config: RVEAConfig, state: RVEAState, fitness: Tensor) -> RVEAState:
    """Store the gen-0 fitness."""
    return replace(state, fit=fitness)


def ask(config: RVEAConfig, state: RVEAState) -> Tuple[Tensor, RVEAState]:
    """Produce one offspring batch: mating pool -> SBX -> polynomial mutation."""
    key, k_mate, k_cross, k_mut = random.split_n(state.key, 4)
    gen = etl.cast(state.gen + 1, etl.int32)

    pop = state.pop
    n_v = pop.shape[0]

    # Mating pool (torch `_mating_pool`): candidates are rows that are not all-NaN.
    valid_mask = enp.logical_not(etl.ops.reduce_max(etl.isnan(pop), axes=1))
    num_valid = etl.cast(etl.sum(etl.cast(valid_mask, etl.int32)), etl.int32)
    mating_pool = randint(k_mate, 0, num_valid, (n_v,), dtype=etl.int32)
    sorted_indices = etl.argsort(
        etl.cast(
            etl.select(valid_mask, enp.arange(n_v, dtype="int32"), 2147483647),
            etl.int32,
        ),
        axis=0,
        stable=True,
    )
    pool = etl.gather(pop, sorted_indices, axis=0)
    mated = etl.gather(pool, mating_pool, axis=0)

    lb = etl.ops.constant(etl.core.tensor(np.asarray(config.lb, dtype=np.float32)))
    ub = etl.ops.constant(etl.core.tensor(np.asarray(config.ub, dtype=np.float32)))
    boundary = enp.stack([lb, ub], axis=0)

    crossover_fn = config.crossover_op if config.crossover_op is not None else simulated_binary
    crossovered = crossover_fn(k_cross, mated)
    mutation_fn = config.mutation_op if config.mutation_op is not None else polynomial_mutation
    offspring = mutation_fn(k_mut, crossovered, boundary)
    offspring = clamp(offspring, lb, ub)
    return offspring, replace(state, key=key, gen=gen)


def tell(config: RVEAConfig, state: RVEAState, offspring: Tensor, fitness: Tensor) -> RVEAState:
    """Merge parents + offspring, RVEA-select n_v survivors, adapt reference vectors."""
    merge_pop = etl.concatenate([state.pop, offspring], axis=0)
    merge_fit = etl.concatenate([state.fit, fitness], axis=0)

    theta = (etl.cast(state.gen, etl.float32) / config.max_gen) ** config.alpha

    selection_fn = config.selection_op if config.selection_op is not None else ref_vec_guided
    survivor, survivor_fit = selection_fn(
        merge_pop, merge_fit, state.reference_vector, theta
    )

    # Reference-vector adaptation (torch `torch.cond`): both branches are cheap
    # and pure, so compute both and select (adapt every `rv_adapt_every` gens).
    max_vals = nanmax(survivor_fit, dim=0)[0]
    min_vals = nanmin(survivor_fit, dim=0)[0]
    adapted = state.init_v * (max_vals - min_vals)
    kept = state.reference_vector
    new_v = etl.select(
        etl.remainder(state.gen, etl.cast(state.rv_adapt_every, etl.int32)) == 0,
        adapted,
        kept,
    )
    return replace(state, pop=survivor, fit=survivor_fit, reference_vector=new_v)
