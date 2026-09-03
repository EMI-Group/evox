"""Functional ETL port of the torch evox SNES algorithm.

Plain functions (`init`/`ask`/`tell`) + frozen config/state dataclasses, per
`src/evox_etl/DESIGN.md`. Torch reference (read-only):
`src/evox/algorithms/so/es_variants/snes.py`.

ETL has no eager mode — these functions may only be called inside an active
trace (`etl.build`/`etl.run`). numpy is used only at trace time to bake the
initial center as a graph constant.
"""

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from etl import core

Tensor = core.Tensor

F32 = np.dtype("float32")


@dataclass(frozen=True)
class SNESConfig:
    """SNES hyperparameters — torch `SNES.__init__` minus `device`."""

    pop_size: int
    center_init: np.ndarray
    sigma: float = 1.0
    lrate_mean: float = 1.0
    temperature: float = 12.5
    weight_type: Literal["recomb", "temp"] = "temp"

    def __post_init__(self) -> None:
        assert self.pop_size > 1
        assert self.weight_type in ("recomb", "temp")
        if isinstance(self.center_init, np.ndarray):
            # Array leaves are illegal inside static args — store a flat tuple.
            object.__setattr__(
                self,
                "center_init",
                tuple(np.asarray(self.center_init, dtype=F32).tolist()),
            )


@dataclass(frozen=True)
class SNESState:
    """SNES mutable state (ETL tensor leaves only)."""

    center: Tensor  # (dim,)
    sigma: Tensor  # (dim,)
    weights: Tensor  # (pop_size, dim)
    noise: Tensor  # (pop_size, dim) — last sampled noise
    best_fitness: Tensor  # scalar f32
    key: Tensor  # RNG key


def _softmax(x: Tensor) -> Tensor:
    """1-D softmax (etl has no softmax op)."""
    m = etl.max(x)
    e = etl.exp(x - m)
    return e / etl.sum(e)


def init(config: SNESConfig, key: Tensor) -> SNESState:
    """Create the initial state (center, sigma, rank weights, zero noise)."""
    dim = len(config.center_init)
    pop_size = config.pop_size
    center = etl.ops.constant(
        etl.core.tensor(np.asarray(config.center_init, dtype=F32))
    )
    sigma = enp.full((dim,), config.sigma, dtype=F32)

    if config.weight_type == "temp":
        ranks = enp.arange(pop_size, dtype=F32) / (pop_size - 1) - 0.5
        w = _softmax(-20.0 * etl.sigmoid(config.temperature * ranks))
    else:  # "recomb"
        w = etl.clamp(
            math.log(pop_size / 2 + 1)
            - etl.log(enp.arange(1, pop_size + 1, dtype=F32)),
            0.0,
            math.inf,
        )
        w = w / etl.sum(w) - 1.0 / pop_size
    weights = etl.tile(enp.expand_dims(w, axis=1), (1, dim))

    noise = enp.zeros((pop_size, dim), dtype=F32)
    best_fitness = enp.full((), np.inf, dtype=F32)
    return SNESState(center, sigma, weights, noise, best_fitness, key)


def ask(config: SNESConfig, state: SNESState) -> tuple[Tensor, SNESState]:
    """Sample a population of `pop_size` candidates from the current Gaussian."""
    pop_size, dim = config.pop_size, len(config.center_init)
    key, subkey = random.split(state.key)
    noise = random.normal(subkey, (pop_size, dim), mean=0.0, std=1.0, dtype=F32)
    population = state.center + noise * etl.reshape(state.sigma, (1, dim))
    return population, replace(state, noise=noise, key=key)


def tell(config: SNESConfig, state: SNESState, fitness: Tensor) -> SNESState:
    """Update center/sigma from the ranked fitness (natural gradient step)."""
    dim = len(config.center_init)
    lrate_sigma = (3 + math.log(dim)) / (5 * math.sqrt(dim))

    order = etl.argsort(fitness)
    sorted_noise = etl.gather(state.noise, order, axis=0)
    grad_mean = etl.sum(state.weights * sorted_noise, axes=0)
    grad_sigma = etl.sum(state.weights * (sorted_noise**2 - 1), axes=0)

    center = state.center + config.lrate_mean * state.sigma * grad_mean
    sigma = state.sigma * etl.exp(lrate_sigma / 2 * grad_sigma)
    best_fitness = etl.minimum(state.best_fitness, etl.min(fitness))
    return replace(state, center=center, sigma=sigma, best_fitness=best_fitness)
