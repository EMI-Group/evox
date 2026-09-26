"""Functional ETL port of ``src/evox/algorithms/so/pso_variants/pso.py``.

Plain (non-defn) functions — ``init``/``init_step``/``step`` — plus the frozen
``PSO`` config and ``PSOState`` state dataclasses.  ``step``/``init_step`` own
one full generation each (candidate generation → ``evaluate`` → state update),
mirroring the torch evox PSO ``step``/``init_step`` 1:1 (see DESIGN.md §4-5).
"""

from dataclasses import dataclass, replace
from typing import Any

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._config_utils import ArrayLike, bake_bounds, normalize_bounds
from evox_etl.operators.jit_fix_operator import clamp

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
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    w: float = 0.6
    phi_p: float = 2.5
    phi_g: float = 0.8


def make_pso(
    pop_size: int,
    lb: ArrayLike,
    ub: ArrayLike,
    w: float = 0.6,
    phi_p: float = 2.5,
    phi_g: float = 0.8,
) -> PSO:
    """Construct a PSO config, normalizing array-like bounds to flat float
    tuples (raises ValueError on non-1-D or shape-mismatched bounds)."""
    lb_t, ub_t = normalize_bounds(lb, ub)
    return PSO(pop_size=pop_size, lb=lb_t, ub=ub_t, w=w, phi_p=phi_p, phi_g=phi_g)


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


def _bounds(config: PSO) -> tuple[Tensor, Tensor]:
    """Bake the (1, dim) lower/upper bound constants from the config arrays."""
    return bake_bounds(config.lb, config.ub, as_row=True)


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


def init_step(config: PSO, state: PSOState, evaluate: Any) -> PSOState:
    """Perform the first step of the PSO optimization.

    Evaluates the initial population via ``evaluate`` and seeds the local and
    global best trackers from it.  See `step` for more details.
    """
    fitness = evaluate(state.pop)
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


def step(config: PSO, state: PSOState, evaluate: Any) -> PSOState:
    """Perform a normal optimization step using PSO.

    This function updates the local best positions and fitness values if the
    current fitness beats the recorded ones, determines the global best via
    ``min_by``, and then adjusts the velocity and positions of the particles
    based on inertia, cognitive, and social components, clamping both within
    the specified bounds.  The proposed population is evaluated with
    ``evaluate`` and its fitness is recorded, completing one full generation.
    """
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
    intermediate = PSOState(
        pop=pop,
        velocity=velocity,
        fit=state.fit,
        local_best_location=local_best_location,
        local_best_fit=local_best_fit,
        global_best_location=global_best_location,
        global_best_fit=global_best_fit,
        key=key,
    )
    fitness = evaluate(pop)
    return replace(intermediate, fit=fitness)
