"""Functional ETL port of the torch DMS-PSO-EL algorithm.

Reference (READ-ONLY, semantics ported 1:1 including quirks):
`src/evox/algorithms/so/pso_variants/dms_pso_el.py`.  Dynamic sub-swarms
regroup every `regrouped_iteration_num` steps and switch to a global-best
strategy once `iteration >= 0.9 * max_iteration` (data-dependent, so the
torch Python `if`s become `etl.cond`).
"""

from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Tuple, Union

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._operator_shims import clamp

Tensor = etl.SymbolicTensor


@dataclass(frozen=True)
class DMSPSOEL:
    """Config of the DMSPSOEL algorithm (torch ``__init__`` minus ``device``).

    ``lb``/``ub`` accept numpy arrays (or any array-like of floats) but are
    stored as tuples of Python floats: the config flows through
    ``etl.build``/``etl.run`` as a static pytree and etl's tracer rejects
    numpy-array leaves, while tuples of Python floats are static values.
    """

    lb: Union[np.ndarray, tuple[float, ...]]
    ub: Union[np.ndarray, tuple[float, ...]]
    dynamic_sub_swarm_size: int = 10  # one of the dynamic sub-swarms size
    dynamic_sub_swarms_num: int = 5  # number of dynamic sub-swarms
    following_sub_swarm_size: int = 10  # following sub-swarm size
    regrouped_iteration_num: int = 50  # number of iterations for regrouping
    max_iteration: int = 100  # maximum number of iterations
    inertia_weight: float = 0.7  # w
    pbest_coefficient: float = 1.5  # c_pbest
    lbest_coefficient: float = 1.5  # c_lbest
    rbest_coefficient: float = 1.0  # c_rbest
    gbest_coefficient: float = 1.0  # c_gbest

    def __post_init__(self) -> None:
        for name in ("lb", "ub"):
            object.__setattr__(
                self, name, tuple(float(v) for v in np.asarray(getattr(self, name)).ravel())
            )


@dataclass(frozen=True)
class DMSPSOELState:
    """State of DMSPSOEL; leaves are etl tensors only."""

    pop: Tensor
    velocity: Tensor
    fit: Tensor
    personal_best_location: Tensor
    personal_best_fit: Tensor
    local_best_location: Tensor
    local_best_fit: Tensor
    regional_best_index: Tensor
    global_best_location: Tensor
    global_best_fit: Tensor
    iteration: Tensor
    key: Tensor


def _bounds(config: DMSPSOEL) -> tuple[Tensor, Tensor]:
    """Bake the (1, dim) float32 bound constants once per use (torch `lb[None, :]`)."""
    lb = etl.ops.constant(etl.core.tensor(np.asarray(config.lb, dtype=np.float32)[None, :]))
    ub = etl.ops.constant(etl.core.tensor(np.asarray(config.ub, dtype=np.float32)[None, :]))
    return lb, ub


def _static_sizes(config: DMSPSOEL) -> tuple[int, int, int, int]:
    """Static Python ints: (pop_size, dynamic_size, dssn, dss)."""
    dssn = config.dynamic_sub_swarms_num
    dss = config.dynamic_sub_swarm_size
    dynamic_size = dssn * dss
    return dynamic_size + config.following_sub_swarm_size, dynamic_size, dssn, dss


def init(config: DMSPSOEL, key: Tensor) -> DMSPSOELState:
    """Draw the initial population and velocity (torch ``__init__`` + setup)."""
    lb, ub = _bounds(config)
    dim = len(config.lb)
    pop_size, dynamic_size, dssn, dss = _static_sizes(config)
    length = ub - lb
    key, key_pop = random.split(key)
    key, key_vel = random.split(key)
    pop = length * random.uniform(key_pop, (pop_size, dim), dtype=etl.float32) + lb
    velocity = 2 * length * random.uniform(key_vel, (pop_size, dim), dtype=etl.float32) - length
    local_best_location = enp.reshape(pop[:dynamic_size], (dssn, dss, dim))[:, 0, :]
    return DMSPSOELState(
        pop=pop,
        velocity=velocity,
        fit=enp.zeros((pop_size,), dtype=etl.float32),
        personal_best_location=pop,
        personal_best_fit=enp.full((pop_size,), np.inf, dtype=etl.float32),
        local_best_location=local_best_location,
        local_best_fit=enp.full((dssn,), np.inf, dtype=etl.float32),
        regional_best_index=enp.zeros((config.following_sub_swarm_size,), dtype=etl.int64),
        global_best_location=enp.zeros((dim,), dtype=etl.float32),
        global_best_fit=enp.full((), np.inf, dtype=etl.float32),
        iteration=enp.full((), 0, dtype=etl.int32),
        key=key,
    )


def init_ask(config: DMSPSOEL, state: DMSPSOELState) -> tuple[Tensor, DMSPSOELState]:
    """Generation-0 ask: the initial population is evaluated as-is."""
    return state.pop, state


def init_tell(config: DMSPSOEL, state: DMSPSOELState, fitness: Tensor) -> DMSPSOELState:
    """Generation-0 tell (torch ``init_step``): store fitness, then count the generation."""
    return replace(state, fit=fitness, iteration=etl.cast(state.iteration + 1, etl.int32))


def ask(config: DMSPSOEL, state: DMSPSOELState) -> tuple[Tensor, DMSPSOELState]:
    """Propose the next population (torch ``step`` split at the evaluate call).

    Strategy 1 while ``iteration < 0.9 * max_iteration``, strategy 2 afterwards;
    the iteration counter advances after the update (torch increments before
    the evaluate call, which is the workflow's job here).
    """
    state = etl.cond(
        state.iteration < 0.9 * config.max_iteration,
        partial(_strategy1, config),
        partial(_strategy2, config),
        state,
    )
    state = replace(state, iteration=etl.cast(state.iteration + 1, etl.int32))
    return state.pop, state


def tell(config: DMSPSOEL, state: DMSPSOELState, fitness: Tensor) -> DMSPSOELState:
    """Store the fitness of the proposed population (torch ``step`` evaluate write)."""
    return replace(state, fit=fitness)


def _strategy1(config: DMSPSOEL, state: DMSPSOELState) -> DMSPSOELState:
    """Torch ``_update_strategy_1``: regroup (every R iterations), then the
    pbest / dynamic-swarm / following-swarm velocity update."""
    state = etl.cond(
        etl.equal(etl.remainder(state.iteration, config.regrouped_iteration_num), 0),
        partial(_regroup, config),
        partial(_identity, config),
        state,
    )
    lb, ub = _bounds(config)
    dim = len(config.lb)
    pop_size, dynamic_size, dssn, dss = _static_sizes(config)
    # Update personal_best
    compare = state.personal_best_fit > state.fit
    personal_best_location = etl.select(
        enp.expand_dims(compare, axis=1), state.pop, state.personal_best_location
    )
    personal_best_fit = etl.select(compare, state.fit, state.personal_best_fit)
    # Update dynamic swarms
    dloc = enp.reshape(state.pop[:dynamic_size], (dssn, dss, dim))
    dfit = enp.reshape(state.fit[:dynamic_size], (dssn, dss))
    dvel = enp.reshape(state.velocity[:dynamic_size], (dssn, dss, dim))
    dpbest = enp.reshape(personal_best_location[:dynamic_size], (dssn, dss, dim))
    # Update following swarm
    following_swarm_location = state.pop[dynamic_size:]
    following_swarm_velocity = state.velocity[dynamic_size:]
    following_swarm_pbest = personal_best_location[dynamic_size:]
    # Update local_best (torch index_select(dloc, 1, lbi).diagonal().T -> flat gather)
    local_best_fit = etl.min(dfit, axes=1)
    local_best_index = etl.argmin(dfit, axis=1)
    offsets = etl.arange(dssn, dtype=etl.int64) * dss
    local_best_location = etl.gather(
        enp.reshape(dloc, (dynamic_size, dim)), offsets + local_best_index, axis=0
    )
    # Update regional_best
    regional_best_location = etl.gather(state.pop, state.regional_best_index, axis=0)
    # Draws in torch order; one subkey per draw, advanced key stored below.
    key, key_rand_pbest = random.split(state.key)
    key, key_rand_lbest = random.split(key)
    key, key_rand_rbest = random.split(key)
    rand_pbest = random.uniform(key_rand_pbest, (pop_size, dim), dtype=etl.float32)
    rand_lbest = random.uniform(key_rand_lbest, (dssn, dss, dim), dtype=etl.float32)
    rand_rbest = random.uniform(key_rand_rbest, (config.following_sub_swarm_size, dim), dtype=etl.float32)
    # Calculate Dynamic Swarms Velocity
    dynamic_swarms_rand_pbest = enp.reshape(rand_pbest[:dynamic_size], (dssn, dss, dim))
    dynamic_swarms_velocity = (
        config.inertia_weight * dvel
        + config.pbest_coefficient * dynamic_swarms_rand_pbest * (dpbest - dloc)
        + config.lbest_coefficient
        * rand_lbest
        * (enp.expand_dims(local_best_location, axis=1) - dloc)
    )
    # Calculate Following Swarm Velocity
    following_swarm_rand_pbest = rand_pbest[dynamic_size:]
    following_swarm_velocity = (
        config.inertia_weight * following_swarm_velocity
        + config.pbest_coefficient
        * following_swarm_rand_pbest
        * (following_swarm_pbest - following_swarm_location)
        + config.rbest_coefficient
        * rand_rbest
        * (regional_best_location - following_swarm_location)
    )
    # Update Population
    velocity = etl.concatenate(
        [enp.reshape(dynamic_swarms_velocity, (dynamic_size, dim)), following_swarm_velocity],
        axis=0,
    )
    pop = clamp(state.pop + velocity, lb, ub)
    velocity = clamp(velocity, lb, ub)
    return replace(
        state,
        pop=pop,
        velocity=velocity,
        personal_best_location=personal_best_location,
        personal_best_fit=personal_best_fit,
        local_best_location=local_best_location,
        local_best_fit=local_best_fit,
        key=key,
    )


def _strategy2(config: DMSPSOEL, state: DMSPSOELState) -> DMSPSOELState:
    """Torch ``_update_strategy_2``: pbest update + global-best guided velocity."""
    lb, ub = _bounds(config)
    dim = len(config.lb)
    pop_size = _static_sizes(config)[0]
    # Update personal_best
    compare = state.personal_best_fit > state.fit
    personal_best_location = etl.select(
        enp.expand_dims(compare, axis=1), state.pop, state.personal_best_location
    )
    personal_best_fit = etl.select(compare, state.fit, state.personal_best_fit)
    # Update global_best
    global_best_fit = etl.min(personal_best_fit, axes=0)
    global_best_idx = etl.argmin(personal_best_fit, axis=0)
    global_best_location = enp.reshape(
        etl.gather(personal_best_location, enp.reshape(global_best_idx, (1,)), axis=0),
        (dim,),
    )
    # Draws in torch order; one subkey per draw, advanced key stored below.
    key, key_rand_pbest = random.split(state.key)
    key, key_rand_gbest = random.split(key)
    rand_pbest = random.uniform(key_rand_pbest, (pop_size, dim), dtype=etl.float32)
    rand_gbest = random.uniform(key_rand_gbest, (pop_size, dim), dtype=etl.float32)
    velocity = (
        config.inertia_weight * state.velocity
        + config.pbest_coefficient * rand_pbest * (personal_best_location - state.pop)
        + config.gbest_coefficient * rand_gbest * (global_best_location - state.pop)
    )
    pop = clamp(state.pop + velocity, lb, ub)
    velocity = clamp(velocity, lb, ub)
    return replace(
        state,
        pop=pop,
        velocity=velocity,
        personal_best_location=personal_best_location,
        personal_best_fit=personal_best_fit,
        global_best_location=global_best_location,
        global_best_fit=global_best_fit,
        key=key,
    )


def _regroup(config: DMSPSOEL, state: DMSPSOELState) -> DMSPSOELState:
    """Torch ``_regroup``: randomly permute the dynamic sub-swarm and rebuild
    ``regional_best_index`` from the PRE-regroup dynamic-swarm fit (quirk)."""
    dynamic_size = _static_sizes(config)[1]
    sort_index = etl.argsort(state.fit, axis=0, stable=True)
    key, subkey = random.split(state.key)
    dynamic_swarm_population_index = random.permutation(subkey, dynamic_size, dtype=etl.int64)
    regroup_index = etl.concatenate(
        [dynamic_swarm_population_index, sort_index[dynamic_size:]], axis=0
    )
    pop = etl.gather(state.pop, regroup_index, axis=0)
    velocity = etl.gather(state.velocity, regroup_index, axis=0)
    personal_best_location = etl.gather(state.personal_best_location, regroup_index, axis=0)
    personal_best_fit = etl.gather(state.personal_best_fit, regroup_index, axis=0)
    dynamic_swarm_fit = state.fit[:dynamic_size]
    regional_best_index = etl.argsort(dynamic_swarm_fit, axis=0)[
        : config.following_sub_swarm_size
    ]
    return replace(
        state,
        pop=pop,
        velocity=velocity,
        personal_best_location=personal_best_location,
        personal_best_fit=personal_best_fit,
        regional_best_index=regional_best_index,
        key=key,
    )


def _identity(config: DMSPSOEL, state: DMSPSOELState) -> DMSPSOELState:
    """No-op branch for the regroup ``etl.cond`` (torch skipped ``_regroup``)."""
    return state
