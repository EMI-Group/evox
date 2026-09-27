"""Minimal toy SO algorithm WITHOUT the ``record_step`` aux hook (back-compat).

Same shape as ``aux_toy_algorithm.py`` minus the OPTIONAL ``record_step``
hook: ``StdWorkflow`` therefore detects no aux hook (``_has_record_step`` is
False) and keeps the LEGACY behavior — it appends the monitor's
``latest_solution`` to ``config.pop_history`` each generation.

The config MUST live in its own module (the workflow resolves the algorithm's
``init``/``step`` via ``type(config).__module__``); this module defines no
``record_step`` at all, which is exactly the legacy path under test.

No torch imports — numpy backend only.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import etl
import etl.numpy as enp
import etl.random as random
import numpy as np

__all__ = [
    "AuxToyNoHookConfig",
    "AuxToyNoHookState",
    "init",
    "step",
]

Tensor = etl.SymbolicTensor

F32 = np.dtype("float32")


@dataclasses.dataclass(frozen=True)
class AuxToyNoHookConfig:
    """Frozen hyperparameters of the no-hook toy random-search algorithm."""

    pop_size: int = 8
    dim: int = 3
    sigma: float = 0.1


@dataclasses.dataclass(frozen=True)
class AuxToyNoHookState:
    """Algorithm state: tensor leaves ONLY (shapes are static across steps)."""

    population: Tensor  # (pop_size, dim)
    center: Tensor  # (dim,) mean of the last evaluated candidate batch
    best_fitness: Tensor  # () running minimum
    key: Tensor  # () RNG key


def init(config: AuxToyNoHookConfig, key: Tensor) -> AuxToyNoHookState:
    """Create the initial random population; no fitness is drawn yet."""
    key, subkey = random.split(key)
    population = random.normal(subkey, (config.pop_size, config.dim), dtype=F32)
    center = enp.zeros((config.dim,), dtype=F32)
    best_fitness = enp.full((), np.inf, dtype=F32)
    return AuxToyNoHookState(population=population, center=center, best_fitness=best_fitness, key=key)


def step(
    config: AuxToyNoHookConfig,
    state: AuxToyNoHookState,
    evaluate: Callable[[Tensor], Tensor],
) -> AuxToyNoHookState:
    """Run ONE generation: perturb, evaluate the batch, update the state."""
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
