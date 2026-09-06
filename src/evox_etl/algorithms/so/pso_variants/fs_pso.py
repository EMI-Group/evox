"""Functional ETL port of the torch Feature-Selection PSO algorithm.

Plain functions only (no ``@etl.defn`` — ETL has no eager mode; everything
runs inside an active trace). 1:1 port of the read-only torch reference in
``src/evox/algorithms/so/pso_variants/fs_pso.py``. numpy is used at trace
time to bake constants and on the host in the config constructor.
"""

from dataclasses import dataclass, replace

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from etl import core

from evox_etl.algorithms._config_utils import (
    ArrayLike,
    bake_bounds,
    bake_float32_constant,
    normalize_bounds,
    to_float_tuple,
)
from evox_etl.operators.jit_fix_operator import clamp

from .utils import min_by

Tensor = core.Tensor


def _bake_bounds(config: "FSPSO") -> tuple[Tensor, Tensor]:
    """Bake lb/ub config arrays as (1, dim) float32 graph constants (torch ``lb[None, :]``)."""
    return bake_bounds(config.lb, config.ub, as_row=True)


@dataclass(frozen=True)
class FSPSO:
    """The Feature Selection PSO algorithm."""

    pop_size: int
    lb: np.ndarray
    ub: np.ndarray
    inertia_weight: float = 0.6  # w
    cognitive_coefficient: float = 2.5  # c
    social_coefficient: float = 0.8  # s
    mean: np.ndarray | None = None
    stdev: np.ndarray | None = None
    mutate_rate: float = 0.01  # mutation ratio


def _check_stat(name: str, value: ArrayLike, dim: int) -> tuple[float, ...]:
    """Validate an optional mean/stdev: 1-D, length dim; flatten to a float tuple."""
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D with length {dim} (matching lb), got shape {arr.shape}"
        )
    result = to_float_tuple(arr)
    if len(result) != dim:
        raise ValueError(
            f"{name} must have length {dim} (matching lb), got {len(result)}"
        )
    return result


def make_fs_pso(
    pop_size: int,
    lb: ArrayLike,
    ub: ArrayLike,
    inertia_weight: float = 0.6,
    cognitive_coefficient: float = 2.5,
    social_coefficient: float = 0.8,
    mean: ArrayLike | None = None,
    stdev: ArrayLike | None = None,
    mutate_rate: float = 0.01,
) -> FSPSO:
    """Normalize array-like bounds/stat params to flat float tuples; ValueError on invalid bounds or stat length."""
    lb_t, ub_t = normalize_bounds(lb, ub)
    dim = len(lb_t)
    mean_t = None if mean is None else _check_stat("mean", mean, dim)
    stdev_t = None if stdev is None else _check_stat("stdev", stdev, dim)
    return FSPSO(
        pop_size=pop_size,
        lb=lb_t,
        ub=ub_t,
        inertia_weight=inertia_weight,
        cognitive_coefficient=cognitive_coefficient,
        social_coefficient=social_coefficient,
        mean=mean_t,
        stdev=stdev_t,
        mutate_rate=mutate_rate,
    )


@dataclass(frozen=True)
class FSPSOState:
    """Tensor leaves only (float32 data + int64 key); mirrors the torch evox State fields."""

    pop: Tensor
    fit: Tensor
    velocity: Tensor
    local_best_location: Tensor
    local_best_fit: Tensor
    global_best_location: Tensor
    global_best_fit: Tensor
    key: Tensor


def init(config: FSPSO, key: Tensor) -> FSPSOState:
    """Draw the initial population, velocity and best-trackers."""
    pop_size = config.pop_size
    dim = len(config.lb)
    lb, ub = _bake_bounds(config)
    length = ub - lb
    key, subkey1, subkey2 = random.split_n(key, 3)
    if config.mean is not None and config.stdev is not None:
        mean_c = bake_float32_constant(config.mean, shape=(1, -1))
        stdev_c = bake_float32_constant(config.stdev, shape=(1, -1))
        pop = clamp(
            mean_c + stdev_c * random.normal(subkey1, (pop_size, dim), 0.0, 1.0, etl.float32),
            lb,
            ub,
        )
        velocity = stdev_c * random.normal(subkey2, (pop_size, dim), 0.0, 1.0, etl.float32)
    else:
        pop = length * random.uniform(subkey1, (pop_size, dim), 0.0, 1.0, etl.float32) + lb
        velocity = random.uniform(subkey2, (pop_size, dim), 0.0, 1.0, etl.float32) * length * 2 - length
    return FSPSOState(
        pop=pop,
        fit=enp.zeros((pop_size,), etl.float32),
        velocity=velocity,
        local_best_location=pop,
        local_best_fit=enp.full((pop_size,), float("inf"), dtype=etl.float32),
        global_best_location=pop[0],
        global_best_fit=enp.full((), float("inf"), dtype=etl.float32),
        key=key,
    )


def init_ask(config: FSPSO, state: FSPSOState) -> tuple[Tensor, FSPSOState]:
    """Return the initial population for evaluation."""
    return state.pop, state


def init_tell(config: FSPSO, state: FSPSOState, fitness: Tensor) -> FSPSOState:
    """Record the initial fitness (global best location is NOT updated — torch parity)."""
    return replace(
        state,
        fit=fitness,
        local_best_fit=fitness,
        global_best_fit=etl.min(fitness, axes=0),
    )


def ask(config: FSPSO, state: FSPSOState) -> tuple[Tensor, FSPSOState]:
    """Propose a new population: elite velocity update plus mutated offspring."""
    pop_size = config.pop_size
    half = pop_size // 2
    dim = len(config.lb)
    lb, ub = _bake_bounds(config)

    # ----------------Enhancement----------------
    ranked_index = etl.argsort(state.fit, axis=0, stable=True)
    elite_index = ranked_index[:half]
    elite_pop = etl.gather(state.pop, elite_index, axis=0)
    elite_velocity = etl.gather(state.velocity, elite_index, axis=0)
    elite_fit = etl.gather(state.fit, elite_index, axis=0)
    elite_local_best_location = etl.gather(state.local_best_location, elite_index, axis=0)
    elite_local_best_fit = etl.gather(state.local_best_fit, elite_index, axis=0)

    compare = elite_local_best_fit > elite_fit
    local_best_location = etl.select(
        enp.expand_dims(compare, 1), elite_pop, elite_local_best_location
    )
    local_best_fit = etl.select(compare, elite_fit, elite_local_best_fit)

    global_best_location, global_best_fit = min_by(
        [enp.expand_dims(state.global_best_location, 0), elite_pop],
        [enp.expand_dims(state.global_best_fit, 0), elite_fit],
    )

    key, rg_key, rp_key, t1_key, t2_key, off_key, mp_key = random.split_n(state.key, 7)
    rg = random.uniform(rg_key, (half, dim), 0.0, 1.0, etl.float32)
    rp = random.uniform(rp_key, (half, dim), 0.0, 1.0, etl.float32)
    updated_elite_velocity = (
        config.inertia_weight * elite_velocity
        + config.cognitive_coefficient * rp * (elite_local_best_location - elite_pop)
        + config.social_coefficient * rg * (global_best_location - elite_pop)
    )
    updated_elite_pop = clamp(elite_pop + updated_elite_velocity, lb, ub)
    updated_elite_velocity = clamp(updated_elite_velocity, lb, ub)

    # ----------------Crossover----------------
    tournament1 = random.randint(t1_key, (half,), 0, half, dtype=etl.int32)
    tournament2 = random.randint(t2_key, (half,), 0, half, dtype=etl.int32)
    compare = etl.gather(elite_fit, tournament1, axis=0) < etl.gather(
        elite_fit, tournament2, axis=0
    )
    mutating_pool = etl.select(compare, tournament1, tournament2)

    # Extend (mutate and create new generation)
    original_population = etl.gather(elite_pop, mutating_pool, axis=0)
    offspring_velocity = etl.gather(elite_velocity, mutating_pool, axis=0)

    offset = (2 * random.uniform(off_key, (half, dim), 0.0, 1.0, etl.float32) - 1) * (ub - lb)
    mutation_prob = random.uniform(mp_key, (half, dim), 0.0, 1.0, etl.float32)
    mask = mutation_prob < config.mutate_rate
    offspring_population = original_population + etl.select(mask, offset, 0)
    offspring_population = clamp(offspring_population, lb, ub)
    offspring_local_best_location = offspring_population
    offspring_local_best_fit = enp.full((half,), float("inf"), dtype=etl.float32)

    # Concatenate updated and offspring populations
    pop = etl.concatenate([updated_elite_pop, offspring_population], axis=0)
    velocity = etl.concatenate([updated_elite_velocity, offspring_velocity], axis=0)
    local_best_location = etl.concatenate(
        [local_best_location, offspring_local_best_location], axis=0
    )
    local_best_fit = etl.concatenate([local_best_fit, offspring_local_best_fit], axis=0)

    new_state = FSPSOState(
        pop=pop,
        fit=state.fit,
        velocity=velocity,
        local_best_location=local_best_location,
        local_best_fit=local_best_fit,
        global_best_location=global_best_location,
        global_best_fit=global_best_fit,
        key=key,
    )
    return pop, new_state


def tell(config: FSPSO, state: FSPSOState, fitness: Tensor) -> FSPSOState:
    """Record the evaluated fitness of the proposed population."""
    return replace(state, fit=fitness)
