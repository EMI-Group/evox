"""Functional ETL port of the torch evox DES algorithm.

Plain functions (`init`/`ask`/`tell`) + frozen config/state dataclasses, per
`src/evox_etl/DESIGN.md`. Torch reference (read-only):
`src/evox/algorithms/so/es_variants/des.py`.

ETL has no eager mode — these functions may only be called inside an active
trace (`etl.build`/`etl.run`). numpy is used only at trace time to bake the
initial center as a graph constant.
"""

from dataclasses import dataclass, replace

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from etl import core

from evox_etl.algorithms._config_utils import ArrayLike, require_gt, to_float_tuple

from .snes import _softmax  # shared 1-D softmax, consolidated in snes.py

Tensor = core.Tensor

F32 = np.dtype("float32")


@dataclass(frozen=True)
class DESConfig:
    """DES hyperparameters — torch `DES.__init__` minus `device`.

    Dumb frozen config storing plain static leaves: `center_init` is a flat
    float32 tuple. Array inputs are normalized and validated in `make_des`.
    """

    pop_size: int
    center_init: tuple[float, ...]
    temperature: float = 12.5
    sigma_init: float = 0.1


def make_des(
    pop_size: int,
    center_init: ArrayLike,
    temperature: float = 12.5,
    sigma_init: float = 0.1,
) -> DESConfig:
    """Build a `DESConfig`, normalizing array inputs and validating parameters.

    :param pop_size: Population size; must be > 1.
    :param center_init: Initial center of the population (1-D array-like),
        normalized to a flat float32 tuple.
    :param temperature: Temperature parameter for the softmax. Defaults to 12.5.
    :param sigma_init: Initial standard deviation of the noise. Defaults to 0.1.
    """
    require_gt("pop_size", pop_size, 1)
    return DESConfig(
        pop_size=pop_size,
        center_init=to_float_tuple(center_init, dtype=np.float32),
        temperature=temperature,
        sigma_init=sigma_init,
    )


@dataclass(frozen=True)
class DESState:
    """DES mutable state (ETL tensor leaves only)."""

    center: Tensor  # (dim,)
    sigma: Tensor  # (dim,)
    noise: Tensor  # (pop_size, dim) — last sampled noise
    best_fitness: Tensor  # scalar f32
    key: Tensor  # RNG key


def init(config: DESConfig, key: Tensor) -> DESState:
    """Create the initial state (center, sigma, zero noise)."""
    dim = len(config.center_init)
    pop_size = config.pop_size
    center = etl.ops.constant(
        etl.core.tensor(np.asarray(config.center_init, dtype=F32))
    )
    sigma = enp.full((dim,), config.sigma_init, dtype=F32)
    noise = enp.zeros((pop_size, dim), dtype=F32)
    best_fitness = enp.full((), np.inf, dtype=F32)
    return DESState(center, sigma, noise, best_fitness, key)


def ask(config: DESConfig, state: DESState) -> tuple[Tensor, DESState]:
    """Sample a population of `pop_size` candidates from the current Gaussian."""
    pop_size, dim = config.pop_size, len(config.center_init)
    key, subkey = random.split(state.key)
    noise = random.normal(subkey, (pop_size, dim), mean=0.0, std=1.0, dtype=F32)
    population = state.center + noise * state.sigma
    return population, replace(state, noise=noise, key=key)


def tell(config: DESConfig, state: DESState, fitness: Tensor) -> DESState:
    """Update center/sigma from the ranked population (DES update rule)."""
    pop_size, dim = config.pop_size, len(config.center_init)

    population = state.center + state.noise * state.sigma
    order = etl.argsort(fitness)
    sorted_pop = etl.gather(population, order, axis=0)

    ranks = enp.arange(pop_size, dtype=F32) / (pop_size - 1) - 0.5
    weight = _softmax(-20.0 * etl.sigmoid(config.temperature * ranks))
    weight = etl.tile(enp.expand_dims(weight, axis=1), (1, dim))

    weight_mean = etl.sum(weight * sorted_pop, axes=0)
    weight_sigma = etl.sqrt(
        etl.sum(weight * (sorted_pop - state.center) ** 2, axes=0) + 1e-6
    )

    center = state.center + 1.0 * (weight_mean - state.center)
    sigma = state.sigma + 0.1 * (weight_sigma - state.sigma)
    best_fitness = etl.minimum(state.best_fitness, etl.min(fitness))
    return replace(state, center=center, sigma=sigma, best_fitness=best_fitness)
