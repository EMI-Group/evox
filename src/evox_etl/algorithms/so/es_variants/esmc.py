"""Functional ETL port of the torch evox ESMC algorithm (plain functions).

Reference (read-only): ``src/evox/algorithms/so/es_variants/esmc.py`` (the DES
algorithm from Learn2Hop).  ETL has no eager mode, so ``init``/``ask``/``tell``
are plain functions traced via ``etl.build``/``etl.run``; the torch ``step`` is
split at ``self.evaluate``.  ``pop_size`` must be odd (mirrored sampling).
"""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from etl.core import SymbolicTensor

from evox_etl.algorithms._config_utils import (
    ArrayLike,
    bake_float32_constant,
    require_choice,
    require_gt,
    to_float_tuple,
)

from .adam_step import adam_single_tensor

F32 = np.dtype("float32")


@dataclass(frozen=True)
class ESMCConfig:
    """Hyperparameters of the ESMC algorithm (mirrors the torch ``__init__``).

    Dumb frozen config storing plain static leaves: `center_init` arrives as a
    flat float32 tuple, normalized by `make_esmc` (static config — baked into
    the graph as a constant at init).
    """

    pop_size: int
    center_init: tuple[float, ...]
    optimizer: Literal["adam"] | None = None
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01
    lr: float = 0.05
    sigma: float = 0.03


def make_esmc(
    pop_size: int,
    center_init: ArrayLike,
    optimizer: Literal["adam"] | None = None,
    sigma_decay: float = 1.0,
    sigma_limit: float = 0.01,
    lr: float = 0.05,
    sigma: float = 0.03,
) -> ESMCConfig:
    """Build an `ESMCConfig`, normalizing `center_init` and validating fields."""
    require_gt("pop_size", pop_size, 1)
    if pop_size % 2 != 1:
        raise ValueError(f"pop_size must be odd (mirrored sampling), got {pop_size!r}")
    require_choice("optimizer", optimizer, (None, "adam"))
    return ESMCConfig(
        pop_size=pop_size,
        center_init=to_float_tuple(center_init, dtype=np.float32),
        optimizer=optimizer,
        sigma_decay=sigma_decay,
        sigma_limit=sigma_limit,
        lr=lr,
        sigma=sigma,
    )


@dataclass(frozen=True)
class ESMCState:
    """Mutable state of ESMC; all leaves are etl tensors."""

    center: SymbolicTensor
    sigma: SymbolicTensor
    z: SymbolicTensor
    exp_avg: SymbolicTensor
    exp_avg_sq: SymbolicTensor
    best_fitness: SymbolicTensor
    key: SymbolicTensor


def init(config: ESMCConfig, key: SymbolicTensor) -> ESMCState:
    """Create the initial state; ``center`` is baked as a graph constant."""
    dim = len(config.center_init)
    center = bake_float32_constant(config.center_init)
    sigma = enp.full((dim,), config.sigma, dtype=F32)
    z = enp.zeros((config.pop_size, dim), dtype=F32)
    exp_avg = enp.zeros((dim,), dtype=F32)
    exp_avg_sq = enp.zeros((dim,), dtype=F32)
    best_fitness = enp.full((), np.inf, dtype=F32)
    return ESMCState(center, sigma, z, exp_avg, exp_avg_sq, best_fitness, key)


def ask(config: ESMCConfig, state: ESMCState) -> tuple[SymbolicTensor, ESMCState]:
    """Sample the mirrored population [0; z_plus; -z_plus] around the center."""
    dim = len(config.center_init)
    key, subkey = random.split(state.key)
    z_plus = random.normal(
        subkey, (config.pop_size // 2, dim), mean=0.0, std=1.0, dtype=F32
    )
    z = etl.concatenate(
        [enp.zeros((1, dim), dtype=F32), z_plus, -1.0 * z_plus], axis=0
    )
    population = state.center + z * enp.reshape(state.sigma, (1, dim))
    return population, replace(state, z=z, key=key)


def tell(
    config: ESMCConfig, state: ESMCState, fitness: SymbolicTensor
) -> ESMCState:
    """Update the center (SGD or adam) and decay sigma from the mirrored pairs."""
    half = (config.pop_size - 1) // 2
    bline = fitness[0]
    noise_1 = state.z[1 : half + 1]
    fit_diff = etl.minimum(fitness[1 : half + 1], bline) - etl.minimum(
        fitness[half + 1 :], bline
    )
    fit_diff_noise = etl.dot(enp.expand_dims(fit_diff, axis=0), noise_1)[0]
    theta_grad = 1.0 / half * fit_diff_noise
    if config.optimizer == "adam":
        center, exp_avg, exp_avg_sq = adam_single_tensor(
            state.center,
            theta_grad,
            state.exp_avg,
            state.exp_avg_sq,
            beta1=0.9,
            beta2=0.999,
            lr=config.lr,
        )
    else:
        center = state.center - config.lr * theta_grad
        exp_avg, exp_avg_sq = state.exp_avg, state.exp_avg_sq
    sigma = etl.maximum(state.sigma * config.sigma_decay, config.sigma_limit)
    best_fitness = etl.minimum(state.best_fitness, etl.min(fitness))
    return replace(
        state,
        center=center,
        sigma=sigma,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        best_fitness=best_fitness,
    )
