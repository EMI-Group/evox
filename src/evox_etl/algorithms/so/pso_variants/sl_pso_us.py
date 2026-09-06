"""Functional ETL port of the torch SLPSOUS algorithm.

Social Learning Particle Swarm Optimization with Uniform Sampling for
demonstrator choice (SLPSOUS). Ported 1:1 from the torch reference
``src/evox/algorithms/so/pso_variants/sl_pso_us.py``.
"""

from dataclasses import dataclass, replace
from typing import Any

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._config_utils import ArrayLike, bake_bounds, normalize_bounds
from evox_etl.operators.jit_fix_operator import clamp, clamp_int

from .utils import min_by

__all__ = [
    "SLPSOUS",
    "SLPSOUSState",
    "make_sl_pso_us",
    "init",
    "init_ask",
    "init_tell",
    "ask",
    "tell",
]


@dataclass(frozen=True)
class SLPSOUS:
    """Config of the SLPSOUS algorithm (mirrors the torch ``__init__`` minus device)."""

    pop_size: int
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    social_influence_factor: float = 0.2  # epsilon
    demonstrator_choice_factor: float = 0.7  # theta


def make_sl_pso_us(
    pop_size: int,
    lb: ArrayLike,
    ub: ArrayLike,
    social_influence_factor: float = 0.2,
    demonstrator_choice_factor: float = 0.7,
) -> SLPSOUS:
    """Construct an SLPSOUS config from array-like bounds (torch signature minus device).

    Normalizes ``lb``/``ub`` to flat tuples of Python floats (the config stores
    plain static leaves per DESIGN.md §4.1). Raises ValueError when a bound is
    not 1-D or the two bounds have mismatched shapes (torch requires 1-D).
    """
    lb_t, ub_t = normalize_bounds(lb, ub)
    return SLPSOUS(
        pop_size=pop_size,
        lb=lb_t,
        ub=ub_t,
        social_influence_factor=social_influence_factor,
        demonstrator_choice_factor=demonstrator_choice_factor,
    )


@dataclass(frozen=True)
class SLPSOUSState:
    """Mutable state of SLPSOUS (all leaves are etl tensors)."""

    pop: Any
    fit: Any
    velocity: Any
    global_best_location: Any
    global_best_fit: Any
    key: Any


def _bake_bounds(config: SLPSOUS) -> tuple[Any, Any]:
    """Bake the (1, dim) lower/upper bound row vectors as graph constants."""
    return bake_bounds(config.lb, config.ub, as_row=True)


def init(config: SLPSOUS, key: Any) -> SLPSOUSState:
    """Draw the initial population and velocities; fitness is a placeholder."""
    dim = len(config.lb)
    lb, ub = _bake_bounds(config)
    length = ub - lb
    key, subkey1 = random.split(key)
    pop = length * random.uniform(subkey1, (config.pop_size, dim), 0.0, 1.0, etl.float32) + lb
    key, subkey2 = random.split(key)
    velocity = 2 * length * random.uniform(subkey2, (config.pop_size, dim), 0.0, 1.0, etl.float32) - length
    fit = enp.zeros((config.pop_size,), etl.float32)
    global_best_location = pop[0]
    global_best_fit = enp.full((), float("inf"), etl.float32)
    return SLPSOUSState(
        pop=pop,
        fit=fit,
        velocity=velocity,
        global_best_location=global_best_location,
        global_best_fit=global_best_fit,
        key=key,
    )


def init_ask(config: SLPSOUS, state: SLPSOUSState) -> tuple[Any, SLPSOUSState]:
    """Return the initial population for evaluation."""
    return state.pop, state


def init_tell(config: SLPSOUS, state: SLPSOUSState, fitness: Any) -> SLPSOUSState:
    """Record the initial fitness; update the global best fit (torch init_step 1:1)."""
    return replace(state, fit=fitness, global_best_fit=etl.min(fitness, axes=0))


def ask(config: SLPSOUS, state: SLPSOUSState) -> tuple[Any, SLPSOUSState]:
    """One SLPSOUS step up to the evaluate point; returns (pop, new_state)."""
    dim = len(config.lb)
    lb, ub = _bake_bounds(config)
    global_best_location, global_best_fit = min_by(
        [enp.expand_dims(state.global_best_location, 0), state.pop],
        [enp.expand_dims(state.global_best_fit, 0), state.fit],
    )
    # Demonstrator choice (uniform sampling): sort worst to best, then each
    # individual picks a demonstrator uniformly from the range [q, pop_size].
    ranked_population = etl.gather(
        state.pop, etl.argsort(-state.fit, axis=0, stable=True), axis=0
    )
    q = clamp_int(
        config.pop_size
        - etl.ceil(
            etl.cast(
                config.demonstrator_choice_factor
                * (config.pop_size - (etl.arange(config.pop_size, dtype=etl.int64) + 1) - 1),
                etl.float32,
            )
        ),
        1,
        config.pop_size,
    )
    key, subkey1 = random.split(state.key)
    uniform_distribution = (
        random.uniform(subkey1, (config.pop_size,), 0.0, 1.0, etl.float32)
        * (config.pop_size + 1 - q)
        + q
    )
    index_k = clamp_int(
        etl.cast(etl.floor(uniform_distribution), etl.int64) - 1, 0, config.pop_size - 1
    )
    X_k = etl.gather(ranked_population, index_k, axis=0)
    # Update population and velocity.
    X_avg = etl.mean(state.pop, axes=0)
    key, subkey2 = random.split(key)
    r1 = random.uniform(subkey2, (config.pop_size, dim), 0.0, 1.0, etl.float32)
    key, subkey3 = random.split(key)
    r2 = random.uniform(subkey3, (config.pop_size, dim), 0.0, 1.0, etl.float32)
    key, subkey4 = random.split(key)
    r3 = random.uniform(subkey4, (config.pop_size, dim), 0.0, 1.0, etl.float32)
    velocity = (
        r1 * state.velocity
        + r2 * (X_k - state.pop)
        + r3 * config.social_influence_factor * (X_avg - state.pop)
    )
    pop = clamp(state.pop + velocity, lb, ub)
    velocity = clamp(velocity, lb, ub)
    return pop, replace(
        state,
        pop=pop,
        velocity=velocity,
        global_best_location=global_best_location,
        global_best_fit=global_best_fit,
        key=key,
    )


def tell(config: SLPSOUS, state: SLPSOUSState, fitness: Any) -> SLPSOUSState:
    """Record the fitness of the current population."""
    return replace(state, fit=fitness)
