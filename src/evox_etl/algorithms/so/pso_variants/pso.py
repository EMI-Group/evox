"""Functional ETL port of ``src/evox/algorithms/so/pso_variants/pso.py``.

Plain (non-defn) functions — ``init``/``init_ask``/``init_tell``/``ask``/
``tell`` — plus the frozen ``PSO`` config and ``PSOState`` state dataclasses.
Semantics mirror the torch evox PSO algorithm 1:1 (see DESIGN.md §4-5).
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._operator_shims import clamp

from .utils import min_by

Tensor = etl.SymbolicTensor


@dataclass(frozen=True)
class PSO:
    """Config of the basic PSO algorithm (mirrors torch ``PSO.__init__``).

    :param pop_size: The size of the population.
    :param lb: The lower bounds of the particle positions. Must be a 1-D array.
    :param ub: The upper bounds of the particle positions. Must be a 1-D array.
    :param w: The inertia weight. Defaults to 0.6.
    :param phi_p: The cognitive weight. Defaults to 2.5.
    :param phi_g: The social weight. Defaults to 0.8.
    """

    pop_size: int
    lb: np.ndarray
    ub: np.ndarray
    w: float = 0.6
    phi_p: float = 2.5
    phi_g: float = 0.8

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
class PSOState:
    """State of the PSO algorithm (mirrors the torch ``Mutable`` attributes)."""

    pop: Tensor
    velocity: Tensor
    fit: Tensor
    local_best_location: Tensor
    local_best_fit: Tensor
    global_best_location: Tensor
    global_best_fit: Tensor
    key: Tensor


def _bounds(config: PSO) -> Tuple[Tensor, Tensor]:
    """Bake the (1, dim) lower/upper bound constants from the config arrays."""
    lb = etl.ops.constant(
        etl.core.tensor(np.asarray(config.lb, dtype=np.float32)[None, :])
    )
    ub = etl.ops.constant(
        etl.core.tensor(np.asarray(config.ub, dtype=np.float32)[None, :])
    )
    return lb, ub


def init(config: PSO, key: Tensor) -> PSOState:
    """Draw the initial PSO state (torch ``__init__`` semantics, key-first RNG)."""
    key, subkey = random.split(key)
    lb, ub = _bounds(config)
    length = ub - lb
    pop_size, dim = config.pop_size, len(config.lb)
    subkey_pop, subkey_vel = random.split(subkey)
    pop = (
        length * random.uniform(subkey_pop, (pop_size, dim), 0.0, 1.0, etl.float32)
        + lb
    )
    velocity = (
        2
        * length
        * random.uniform(subkey_vel, (pop_size, dim), 0.0, 1.0, etl.float32)
        - length
    )
    inf_fit = enp.full((pop_size,), float("inf"), dtype=etl.float32)
    return PSOState(
        pop=pop,
        velocity=velocity,
        fit=inf_fit,
        local_best_location=pop,
        local_best_fit=inf_fit,
        global_best_location=pop[0],
        global_best_fit=enp.full((), float("inf"), dtype=etl.float32),
        key=key,
    )


def init_ask(config: PSO, state: PSOState) -> Tuple[Tensor, PSOState]:
    """First-generation ask: expose the initial population (nothing changes)."""
    return state.pop, state


def init_tell(config: PSO, state: PSOState, fitness: Tensor) -> PSOState:
    """First-generation tell (torch ``init_step``): record fitness and bests."""
    global_best_location, global_best_fit = min_by([state.pop], [fitness])
    return PSOState(
        pop=state.pop,
        velocity=state.velocity,
        fit=fitness,
        local_best_location=state.local_best_location,
        local_best_fit=fitness,
        global_best_location=global_best_location,
        global_best_fit=global_best_fit,
        key=state.key,
    )


def ask(config: PSO, state: PSOState) -> Tuple[Tensor, PSOState]:
    """One PSO step up to evaluation (torch ``step`` before ``evaluate``)."""
    lb, ub = _bounds(config)
    pop_size, dim = config.pop_size, len(config.lb)

    compare = state.local_best_fit > state.fit
    local_best_location = etl.select(
        enp.expand_dims(compare, axis=1), state.pop, state.local_best_location
    )
    local_best_fit = etl.select(compare, state.fit, state.local_best_fit)
    global_best_location, global_best_fit = min_by(
        [enp.expand_dims(state.global_best_location, axis=0), state.pop],
        [enp.expand_dims(state.global_best_fit, axis=0), state.fit],
    )
    key, subkey = random.split(state.key)
    subkey_rg, subkey_rp = random.split(subkey)
    rg = random.uniform(subkey_rg, (pop_size, dim), 0.0, 1.0, etl.float32)
    rp = random.uniform(subkey_rp, (pop_size, dim), 0.0, 1.0, etl.float32)
    velocity = (
        config.w * state.velocity
        + config.phi_p * rp * (local_best_location - state.pop)
        + config.phi_g * rg * (global_best_location - state.pop)
    )
    pop = clamp(state.pop + velocity, lb, ub)
    velocity = clamp(velocity, lb, ub)
    return pop, PSOState(
        pop=pop,
        velocity=velocity,
        fit=state.fit,
        local_best_location=local_best_location,
        local_best_fit=local_best_fit,
        global_best_location=global_best_location,
        global_best_fit=global_best_fit,
        key=key,
    )


def tell(config: PSO, state: PSOState, fitness: Tensor) -> PSOState:
    """Record the evaluated fitness (torch ``step`` after ``evaluate``)."""
    return PSOState(
        pop=state.pop,
        velocity=state.velocity,
        fit=fitness,
        local_best_location=state.local_best_location,
        local_best_fit=state.local_best_fit,
        global_best_location=state.global_best_location,
        global_best_fit=state.global_best_fit,
        key=state.key,
    )
