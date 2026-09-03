"""Functional ETL port of the torch SLPSOUS algorithm.

Social Learning Particle Swarm Optimization with Uniform Sampling for
demonstrator choice (SLPSOUS). Ported 1:1 from the torch reference
``src/evox/algorithms/so/pso_variants/sl_pso_us.py``.
"""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.operators.jit_fix_operator import clamp, clamp_int

from .utils import min_by

__all__ = [
    "SLPSOUS",
    "SLPSOUSState",
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
    lb: np.ndarray
    ub: np.ndarray
    social_influence_factor: float = 0.2  # epsilon
    demonstrator_choice_factor: float = 0.7  # theta

    def __post_init__(self) -> None:
        # etl static trace arguments reject numpy arrays/scalars (TraceError);
        # store the bounds as float tuples so this frozen config is a legal
        # static pytree. The constructor API (numpy arrays in) is unchanged.
        lb = np.asarray(self.lb)
        ub = np.asarray(self.ub)
        assert lb.ndim == 1 and ub.ndim == 1 and lb.shape == ub.shape
        object.__setattr__(self, "lb", tuple(float(v) for v in lb))
        object.__setattr__(self, "ub", tuple(float(v) for v in ub))


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
    lb = etl.ops.constant(etl.core.tensor(np.asarray(config.lb, dtype=np.float32)[None, :]))
    ub = etl.ops.constant(etl.core.tensor(np.asarray(config.ub, dtype=np.float32)[None, :]))
    return lb, ub


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
