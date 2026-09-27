"""Minimal toy SO algorithm WITH the optional ``record_step`` aux hook.

The workflow resolves a component's plain functions via
``importlib.import_module(type(config).__module__)`` (the functional module
convention), so this algorithm config must live in its OWN module — the toy
problem and the no-hook toy algorithm live in distinct sibling modules.

Protocol implemented here (see ``evox_etl/core/algorithm.py``):

* ``init(config, key) -> AuxToyAlgorithmState``
* ``step(config, state, evaluate) -> AuxToyAlgorithmState`` — owns ONE
  generation: perturb the population, call ``evaluate(candidates)`` for the
  fitness, then update the state.
* ``record_step(config, state, candidate, fitness) -> dict[str, etl_tensor]`` —
  the OPTIONAL plain host-side auxiliary-history hook ``StdWorkflow`` calls
  after each generation. It receives the post-step algorithm state plus the
  CONCRETE etl tensors ``state.monitor_state.latest_solution``/``latest_fitness``
  and returns a dict of concrete etl tensors (each supporting
  ``.to(Device("cpu")).numpy()``).

No torch imports — numpy backend only.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import etl
import etl.numpy as enp
import etl.random as random
import numpy as np

__all__ = [
    "AuxToyAlgorithmConfig",
    "AuxToyAlgorithmState",
    "init",
    "step",
    "record_step",
]

Tensor = etl.SymbolicTensor

F32 = np.dtype("float32")


@dataclasses.dataclass(frozen=True)
class AuxToyAlgorithmConfig:
    """Frozen hyperparameters of the toy random-search algorithm."""

    pop_size: int = 8
    dim: int = 3
    sigma: float = 0.1


@dataclasses.dataclass(frozen=True)
class AuxToyAlgorithmState:
    """Algorithm state: tensor leaves ONLY (shapes are static across steps)."""

    population: Tensor  # (pop_size, dim)
    center: Tensor  # (dim,) mean of the last evaluated candidate batch
    best_fitness: Tensor  # () running minimum
    key: Tensor  # () RNG key


def init(config: AuxToyAlgorithmConfig, key: Tensor) -> AuxToyAlgorithmState:
    """Create the initial random population; no fitness is drawn yet."""
    key, subkey = random.split(key)
    population = random.normal(subkey, (config.pop_size, config.dim), dtype=F32)
    center = enp.zeros((config.dim,), dtype=F32)
    best_fitness = enp.full((), np.inf, dtype=F32)
    return AuxToyAlgorithmState(population=population, center=center, best_fitness=best_fitness, key=key)


def step(
    config: AuxToyAlgorithmConfig,
    state: AuxToyAlgorithmState,
    evaluate: Callable[[Tensor], Tensor],
) -> AuxToyAlgorithmState:
    """Run ONE generation: perturb, evaluate the batch, update the state.

    ``evaluate`` is the workflow-injected traced closure (opaque, minimization
    semantics); it is called with a ``(pop_size, dim)`` candidate batch and
    returns rank-1 fitness of shape ``(pop_size,)``.
    """
    key, subkey = random.split(state.key)
    noise = random.normal(subkey, (config.pop_size, config.dim), dtype=F32)
    candidates = state.population + config.sigma * noise
    fitness = evaluate(candidates)
    center = enp.mean(candidates, axis=0)
    best_fitness = etl.minimum(state.best_fitness, etl.min(fitness))
    return dataclasses.replace(
        state,
        population=candidates,
        center=center,
        best_fitness=best_fitness,
        key=key,
    )


def record_step(
    config: AuxToyAlgorithmConfig,
    state: AuxToyAlgorithmState,
    candidate: Any,
    fitness: Any,
) -> dict[str, Any]:
    """Host-side aux-history hook: report three tensors for this generation.

    ``candidate``/``fitness`` are the concrete etl tensors the monitor stored as
    ``latest_solution``/``latest_fitness`` (or ``None`` when the monitor state
    lacks them). ``state`` is the POST-step algorithm state. All returned values
    are concrete etl tensors.
    """
    aux: dict[str, Any] = {"center": state.center}
    if candidate is not None:
        aux["pop"] = candidate
    if fitness is not None:
        aux["fit"] = fitness
    return aux
