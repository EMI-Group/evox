"""Functional port of the torch NSGA2 (`src/evox/algorithms/mo/nsga2.py`).

1:1 semantic port in step form (DESIGN.md §4-5): `init` draws the population,
`init_step` owns the full-population first generation, and `step` owns one
whole generation (tournament selection → SBX → polynomial mutation →
`evaluate` → environmental selection). The candidates are evaluated inside
the same call through the opaque `evaluate` closure; the offspring batch is
recorded in `state.offspring` and merged with the parents, keeping the best
`pop_size` via non-dominated rank + crowding distance.

State fields: pop/fit/rank/dis (float32, rank int32), `offspring` (float32,
present from init so the pytree structure/shapes never change), `key` (as
returned by `random.split`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from evox_etl.algorithms._config_utils import ArrayLike, bake_bounds, normalize_bounds
from evox_etl.operators.jit_fix_operator import clamp
from evox_etl.operators.crossover import simulated_binary
from evox_etl.operators.mutation import polynomial_mutation
from evox_etl.operators.selection import (
    nd_environmental_selection,
    tournament_selection_multifit,
)

F32 = np.dtype("float32")
I32 = np.dtype("int32")

__all__ = [
    "NSGA2Config",
    "NSGA2State",
    "make_nsga2",
    "init",
    "init_step",
    "step",
]


@dataclass(frozen=True)
class NSGA2Config:
    """NSGA2 hyperparameters (mirrors the torch ``__init__`` minus device).

    The optional op fields mirror the torch class: ``None`` means the torch
    default (``tournament_selection_multifit`` / ``simulated_binary`` /
    ``polynomial_mutation``). A custom op must match the etl keyed signature of
    the default it replaces — ``selection_op(key, n_parents, fitness)``,
    ``crossover_op(key, x)``, ``mutation_op(key, x, lb, ub)``.
    """

    pop_size: int
    n_objs: int
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    selection_op: Optional[Callable] = None
    mutation_op: Optional[Callable] = None
    crossover_op: Optional[Callable] = None


def _config_flatten(config: NSGA2Config):
    """Zero-child flattening: the config travels as one opaque static node."""
    return [], config


def _config_unflatten(config: NSGA2Config, _children) -> NSGA2Config:
    return config


# This registration is REQUIRED: the config carries optional callable op fields
# (selection_op/mutation_op/crossover_op), and functions are not valid static
# pytree leaves — so the config travels as one opaque childless node through
# etl.build/etl.run untouched.
etl.register_pytree_node(NSGA2Config, _config_flatten, _config_unflatten)


def make_nsga2(
    pop_size: int,
    n_objs: int,
    lb: ArrayLike,
    ub: ArrayLike,
    selection_op: Optional[Callable] = None,
    mutation_op: Optional[Callable] = None,
    crossover_op: Optional[Callable] = None,
) -> NSGA2Config:
    """Construct an NSGA2Config from array-like bounds and optional custom ops.

    Bounds are normalized to flat tuples of plain Python floats; a ``None`` op
    field means the torch default (tournament_selection_multifit,
    simulated_binary, polynomial_mutation). Raises ValueError when lb/ub are not
    1-D, their shapes differ, or an op field is neither None nor callable.
    """
    for name, op in (
        ("selection_op", selection_op),
        ("mutation_op", mutation_op),
        ("crossover_op", crossover_op),
    ):
        if op is not None and not callable(op):
            raise ValueError(f"{name} must be callable or None, got {op!r}")
    lb, ub = normalize_bounds(lb, ub)
    return NSGA2Config(
        pop_size=pop_size,
        n_objs=n_objs,
        lb=lb,
        ub=ub,
        selection_op=selection_op,
        mutation_op=mutation_op,
        crossover_op=crossover_op,
    )


@dataclass(frozen=True)
class NSGA2State:
    """NSGA2 runtime state; tensor leaves only.

    ``offspring`` carries the latest candidate batch from the generation
    stage of ``step`` into its selection stage (both share one call); it
    always has shape (pop_size, dim) so the state pytree never changes.
    """

    pop: etl.SymbolicTensor
    fit: etl.SymbolicTensor
    rank: etl.SymbolicTensor
    dis: etl.SymbolicTensor
    offspring: etl.SymbolicTensor
    key: etl.SymbolicTensor


def init(config: NSGA2Config, key: etl.SymbolicTensor) -> NSGA2State:
    """Draw the initial population uniformly in [lb, ub]; unfit until init_step."""
    lb, ub = bake_bounds(config.lb, config.ub)
    dim = len(config.lb)
    key, subkey = random.split(key)
    pop = random.uniform(subkey, (config.pop_size, dim), 0.0, 1.0, "float32")
    pop = (ub - lb) * pop + lb
    fit = enp.full((config.pop_size, config.n_objs), float("inf"), dtype=F32)
    rank = enp.full((config.pop_size,), np.iinfo(np.int32).max, dtype=I32)
    dis = enp.full((config.pop_size,), float("-inf"), dtype=F32)
    return NSGA2State(pop=pop, fit=fit, rank=rank, dis=dis, offspring=pop, key=key)


def init_step(
    config: NSGA2Config, state: NSGA2State, evaluate: Callable
) -> NSGA2State:
    """First generation: evaluate the FULL initial population, then keep the
    best pop_size of it via non-dominated rank + crowding distance (no RNG)."""
    fitness = evaluate(state.pop)
    pop, fit, rank, dis = nd_environmental_selection(state.pop, fitness, config.pop_size)
    return replace(state, pop=pop, fit=fit, rank=rank, dis=dis)


def step(config: NSGA2Config, state: NSGA2State, evaluate: Callable) -> NSGA2State:
    """Run ONE full generation: tournament -> SBX -> polynomial mutation ->
    ``evaluate`` the offspring -> merge parents + offspring (2*pop_size) and
    select pop_size survivors via non-dominated rank + crowding distance."""
    lb, ub = bake_bounds(config.lb, config.ub)
    selection = config.selection_op or tournament_selection_multifit
    crossover = config.crossover_op or simulated_binary
    mutation = config.mutation_op or polynomial_mutation
    key, k_sel, k_cross, k_mut = random.split_n(state.key, 4)
    mating_pool = selection(
        k_sel,
        config.pop_size,
        [etl.negate(state.dis), etl.cast(state.rank, F32)],
    )
    crossovered = crossover(k_cross, etl.gather(state.pop, mating_pool, axis=0))
    offspring = mutation(k_mut, crossovered, lb, ub)
    offspring = clamp(offspring, lb, ub)
    intermediate = replace(state, offspring=offspring, key=key)

    fitness = evaluate(offspring)
    merge_pop = etl.concatenate([intermediate.pop, intermediate.offspring], axis=0)
    merge_fit = etl.concatenate([intermediate.fit, fitness], axis=0)
    pop, fit, rank, dis = nd_environmental_selection(merge_pop, merge_fit, config.pop_size)
    return replace(intermediate, pop=pop, fit=fit, rank=rank, dis=dis)
