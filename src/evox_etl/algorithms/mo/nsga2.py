"""Functional port of the torch NSGA2 (`src/evox/algorithms/mo/nsga2.py`).

1:1 semantic port in init/ask/tell form (DESIGN.md §4-5): `init` draws the
population, `init_ask`/`init_tell` handle the full-population first
generation, `ask` produces the offspring batch and stores it in
`state.offspring` (JAX-evoX style), `tell` merges parents + offspring and
keeps the best `pop_size` via non-dominated rank + crowding distance.

State fields: pop/fit/rank/dis (float32, rank int32), `offspring` (float32,
present from init so the pytree structure/shapes never change), `key` (as
returned by `random.split`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from evox_etl.algorithms._operator_shims import (
    clamp,
    nd_environmental_selection,
    polynomial_mutation,
    simulated_binary,
    tournament_selection_multifit,
)

F32 = np.dtype("float32")
I32 = np.dtype("int32")

__all__ = [
    "NSGA2Config",
    "NSGA2State",
    "init",
    "init_ask",
    "init_tell",
    "ask",
    "tell",
]


@dataclass(frozen=True)
class NSGA2Config:
    """NSGA2 hyperparameters (mirrors the torch ``__init__`` minus device).

    The optional op fields are kept for signature parity; the torch defaults
    (tournament_selection_multifit, simulated_binary, polynomial_mutation)
    are always used, matching the torch behavior when they are None.
    """

    pop_size: int
    n_objs: int
    lb: np.ndarray
    ub: np.ndarray
    selection_op: Optional[object] = None
    mutation_op: Optional[object] = None
    crossover_op: Optional[object] = None


def _config_flatten(config: NSGA2Config):
    """Zero-child flattening: the config travels as one opaque static node."""
    return [], config


def _config_unflatten(config: NSGA2Config, _children) -> NSGA2Config:
    return config


# ETL v1 rejects numpy arrays as static pytree leaves (they are neither
# TensorSpecs nor static Python values), so the config (which holds lb/ub as
# ndarrays) is registered as a childless pytree node carrying the whole
# config as its context — it then passes through etl.build/etl.run untouched.
etl.register_pytree_node(NSGA2Config, _config_flatten, _config_unflatten)


@dataclass(frozen=True)
class NSGA2State:
    """NSGA2 runtime state; tensor leaves only.

    ``offspring`` carries the latest candidate batch from `ask` into `tell`
    (the workflow passes only fitness back); it always has shape
    (pop_size, dim) so the state pytree never changes.
    """

    pop: etl.SymbolicTensor
    fit: etl.SymbolicTensor
    rank: etl.SymbolicTensor
    dis: etl.SymbolicTensor
    offspring: etl.SymbolicTensor
    key: etl.SymbolicTensor


def _bounds(config: NSGA2Config):
    """Bake lb/ub numpy arrays as graph constants (float32)."""
    lb = etl.ops.constant(etl.core.tensor(np.asarray(config.lb, dtype=F32)))
    ub = etl.ops.constant(etl.core.tensor(np.asarray(config.ub, dtype=F32)))
    return lb, ub


def init(config: NSGA2Config, key: etl.SymbolicTensor) -> NSGA2State:
    """Draw the initial population uniformly in [lb, ub]; unfit until init_tell."""
    lb, ub = _bounds(config)
    dim = config.lb.shape[0]
    key, subkey = random.split(key)
    pop = random.uniform(subkey, (config.pop_size, dim), 0.0, 1.0, "float32")
    pop = (ub - lb) * pop + lb
    fit = enp.full((config.pop_size, config.n_objs), float("inf"), dtype=F32)
    rank = enp.full((config.pop_size,), np.iinfo(np.int32).max, dtype=I32)
    dis = enp.full((config.pop_size,), float("-inf"), dtype=F32)
    return NSGA2State(pop=pop, fit=fit, rank=rank, dis=dis, offspring=pop, key=key)


def init_ask(config: NSGA2Config, state: NSGA2State):
    """Return the full initial population for evaluation (no RNG)."""
    return state.pop, state


def init_tell(config: NSGA2Config, state: NSGA2State, fitness: etl.SymbolicTensor) -> NSGA2State:
    """Keep the best pop_size of the evaluated initial population (no RNG)."""
    pop, fit, rank, dis = nd_environmental_selection(state.pop, fitness, config.pop_size)
    return replace(state, pop=pop, fit=fit, rank=rank, dis=dis)


def ask(config: NSGA2Config, state: NSGA2State):
    """Generate pop_size offspring: tournament -> SBX -> polynomial mutation."""
    lb, ub = _bounds(config)
    boundary = enp.stack([lb, ub], axis=0)
    key, k_sel, k_cross, k_mut = random.split_n(state.key, 4)
    mating_pool = tournament_selection_multifit(
        k_sel,
        config.pop_size,
        [etl.negate(state.dis), etl.cast(state.rank, F32)],
    )
    crossovered = simulated_binary(k_cross, etl.gather(state.pop, mating_pool, axis=0))
    offspring = polynomial_mutation(k_mut, crossovered, boundary, pro_m=1, dis_m=20)
    offspring = clamp(offspring, lb, ub)
    return offspring, replace(state, offspring=offspring, key=key)


def tell(config: NSGA2Config, state: NSGA2State, fitness: etl.SymbolicTensor) -> NSGA2State:
    """Merge parents + offspring (2*pop_size) and select pop_size survivors (no RNG)."""
    merge_pop = etl.concatenate([state.pop, state.offspring], axis=0)
    merge_fit = etl.concatenate([state.fit, fitness], axis=0)
    pop, fit, rank, dis = nd_environmental_selection(merge_pop, merge_fit, config.pop_size)
    return replace(state, pop=pop, fit=fit, rank=rank, dis=dis)
