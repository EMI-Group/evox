"""Functional ETL port of the torch evox CLPSO algorithm.

Plain-function port of ``src/evox/algorithms/so/pso_variants/clpso.py``
(read-only torch reference; the ask/tell split follows the old JAX evox
decomposition at tag v0.9.0, with torch semantics winning).  ETL has no eager
mode, so these functions only run inside an active trace via
``etl.build``/``etl.run``.

Config note: ``CLPSO`` stores ``lb``/``ub`` as flat float tuples (plain static
pytree leaves); ``make_clpso`` normalizes array-like bounds (see DESIGN.md §4.1).
"""

from dataclasses import dataclass, replace
from typing import Tuple

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._config_utils import (
    ArrayLike,
    bake_float32_constant,
    normalize_bounds,
)
from evox_etl.operators.jit_fix_operator import clamp

from .utils import min_by

Tensor = etl.SymbolicTensor


@dataclass(frozen=True)
class CLPSO:
    """CLPSO hyperparameters (torch ``CLPSO.__init__`` minus ``device``)."""

    pop_size: int
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    inertia_weight: float = 0.5
    const_coefficient: float = 1.5
    learning_probability: float = 0.05


def make_clpso(
    pop_size: int,
    lb: ArrayLike,
    ub: ArrayLike,
    inertia_weight: float = 0.5,
    const_coefficient: float = 1.5,
    learning_probability: float = 0.05,
) -> CLPSO:
    """Construct a CLPSO config, normalizing array-like bounds to flat float
    tuples (raises ValueError on non-1-D or shape-mismatched bounds)."""
    lb_t, ub_t = normalize_bounds(lb, ub)
    return CLPSO(
        pop_size=pop_size,
        lb=lb_t,
        ub=ub_t,
        inertia_weight=inertia_weight,
        const_coefficient=const_coefficient,
        learning_probability=learning_probability,
    )


@dataclass(frozen=True)
class CLPSOState:
    """CLPSO mutable state (one etl tensor per leaf)."""

    pop: Tensor
    fit: Tensor
    velocity: Tensor
    personal_best_location: Tensor
    personal_best_fit: Tensor
    global_best_location: Tensor
    global_best_fit: Tensor
    key: Tensor


def _bake(arr: ArrayLike) -> Tensor:
    """Bake a config bound into a (1, dim) float32 graph constant."""
    return bake_float32_constant(arr, shape=(1, -1))


def init(config: CLPSO, key: Tensor) -> CLPSOState:
    """Draw the initial population, velocity and best-tracking placeholders."""
    pop_size = config.pop_size
    dim = len(config.lb)
    lb = _bake(config.lb)
    ub = _bake(config.ub)
    length = ub - lb

    state_key, init_key = random.split(key)
    key_pop, key_vel = random.split(init_key)

    pop = (
        length * random.uniform(key_pop, (pop_size, dim), 0.0, 1.0, etl.float32) + lb
    )
    velocity = (
        2 * length * random.uniform(key_vel, (pop_size, dim), 0.0, 1.0, etl.float32)
        - length
    )
    return CLPSOState(
        pop=pop,
        fit=enp.zeros((pop_size,), dtype=etl.float32),
        velocity=velocity,
        personal_best_location=pop,
        personal_best_fit=enp.full((pop_size,), float("inf"), dtype=etl.float32),
        global_best_location=pop[0],
        global_best_fit=enp.full((), float("inf"), dtype=etl.float32),
        key=state_key,
    )


def init_ask(config: CLPSO, state: CLPSOState) -> Tuple[Tensor, CLPSOState]:
    """First-generation candidates: the whole population."""
    return state.pop, state


def init_tell(config: CLPSO, state: CLPSOState, fitness: Tensor) -> CLPSOState:
    """Store the first-generation fitness and initialize the best trackers."""
    return replace(
        state,
        fit=fitness,
        personal_best_fit=fitness,
        # torch does NOT update global_best_location in init_step — port 1:1.
        global_best_fit=etl.min(fitness, axes=0),
    )


def ask(config: CLPSO, state: CLPSOState) -> Tuple[Tensor, CLPSOState]:
    """Comprehensive-learning velocity/position update; returns the new pop."""
    pop_size = config.pop_size
    dim = len(config.lb)
    lb = _bake(config.lb)
    ub = _bake(config.ub)

    # Draw order mirrors torch step exactly: coefficient, rand1, rand2, possibility.
    key, subkey = random.split(state.key)
    random_coefficient = random.uniform(subkey, (pop_size, dim), 0.0, 1.0, etl.float32)
    key, subkey = random.split(key)
    rand1_index = random.randint(subkey, (pop_size,), 0, pop_size, dtype=etl.int32)
    key, subkey = random.split(key)
    rand2_index = random.randint(subkey, (pop_size,), 0, pop_size, dtype=etl.int32)
    key, subkey = random.split(key)
    rand_possibility = random.uniform(subkey, (pop_size,), 0.0, 1.0, etl.float32)

    learning_index = etl.select(
        etl.gather(state.personal_best_fit, rand1_index, axis=0)
        < etl.gather(state.personal_best_fit, rand2_index, axis=0),
        rand1_index,
        rand2_index,
    )
    # Update personal_best.
    compare = state.personal_best_fit > state.fit
    personal_best_location = etl.select(
        enp.reshape(compare, (pop_size, 1)), state.pop, state.personal_best_location
    )
    personal_best_fit = etl.select(compare, state.fit, state.personal_best_fit)
    # Update global_best.
    global_best_location, global_best_fit = min_by(
        [enp.reshape(state.global_best_location, (1, dim)), state.pop],
        [enp.reshape(state.global_best_fit, (1,)), state.fit],
    )
    # Choose personal_best.
    learning_personal_best = etl.gather(
        personal_best_location, learning_index, axis=0
    )
    personal_best = etl.select(
        enp.reshape(rand_possibility < config.learning_probability, (pop_size, 1)),
        learning_personal_best,
        personal_best_location,
    )
    # Update velocity and position.
    velocity = (
        config.inertia_weight * state.velocity
        + config.const_coefficient * random_coefficient * (personal_best - state.pop)
    )
    velocity = clamp(velocity, lb, ub)
    pop = clamp(state.pop + velocity, lb, ub)

    return pop, replace(
        state,
        pop=pop,
        velocity=velocity,
        personal_best_location=personal_best_location,
        personal_best_fit=personal_best_fit,
        global_best_location=global_best_location,
        global_best_fit=global_best_fit,
        key=key,
    )


def tell(config: CLPSO, state: CLPSOState, fitness: Tensor) -> CLPSOState:
    """Store the evaluated population fitness."""
    return replace(state, fit=fitness)
