"""Functional ETL port of the torch evox ARS algorithm.

Plain (non-`@etl.defn`) functions: they may only be called inside an active
trace (a function passed to `etl.build`/`etl.run`), since ETL has no eager
mode.  Semantics mirror the torch original in
`src/evox/algorithms/so/es_variants/ars.py` exactly.

ARS is described in "Simple random search provides a competitive approach to
reinforcement learning" (https://arxiv.org/abs/1803.07055); this
implementation follows the evosax version like the torch reference.
"""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from .adam_step import adam_single_tensor

Tensor = etl.SymbolicTensor

F32 = np.dtype("float32")


@dataclass(frozen=True)
class ARSConfig:
    """Frozen hyperparameters (torch `ARS.__init__` minus `device`).

    ``center_init`` may be passed as an ``np.ndarray``; it is normalized to a
    flat tuple of float32 values (hashable static config arg).
    """

    pop_size: int
    center_init: tuple[float, ...]
    elite_ratio: float = 0.1
    lr: float = 0.05
    sigma: float = 0.03
    optimizer: Literal["adam"] | None = None

    def __post_init__(self) -> None:
        assert self.pop_size > 1
        assert 0 <= self.elite_ratio <= 1
        if isinstance(self.center_init, np.ndarray):
            object.__setattr__(
                self,
                "center_init",
                tuple(np.asarray(self.center_init, dtype=np.float32).tolist()),
            )


@dataclass(frozen=True)
class ARSState:
    """Algorithm state: tensor leaves only."""

    center: Tensor  # (dim,) current search center
    noise: Tensor  # (pop_size, dim) mirrored noise sampled by the last ask
    exp_avg: Tensor  # (dim,) Adam first moment (always carried)
    exp_avg_sq: Tensor  # (dim,) Adam second moment (always carried)
    best_fitness: Tensor  # () best fitness seen so far
    key: Tensor  # () RNG key


def init(config: ARSConfig, key: Tensor) -> ARSState:
    """Create the initial state; no randomness is drawn (key passed through)."""
    dim = len(config.center_init)
    center = etl.ops.constant(
        etl.core.tensor(np.asarray(config.center_init, dtype=np.float32))
    )
    noise = enp.zeros((config.pop_size, dim), dtype=F32)
    exp_avg = enp.zeros((dim,), dtype=F32)
    exp_avg_sq = enp.zeros((dim,), dtype=F32)
    best_fitness = enp.full((), np.inf, dtype=F32)
    return ARSState(
        center=center,
        noise=noise,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        best_fitness=best_fitness,
        key=key,
    )


def ask(config: ARSConfig, state: ARSState) -> tuple[Tensor, ARSState]:
    """Sample the mirrored population ``center +- sigma * z``."""
    dim = len(config.center_init)
    key, subkey = random.split(state.key)
    half = config.pop_size // 2
    z_plus = random.normal(subkey, (half, dim), dtype=F32)
    noise = etl.concatenate([z_plus, -1.0 * z_plus], axis=0)
    population = state.center + config.sigma * noise
    return population, replace(state, noise=noise, key=key)


def tell(config: ARSConfig, state: ARSState, fitness: Tensor) -> ARSState:
    """Rank mirrored pairs, keep the elite ones, and step the center."""
    half = config.pop_size // 2
    elite_pop_size = max(1, int(half * config.elite_ratio))
    fit_1 = fitness[:half]
    fit_2 = fitness[half:]
    elite_idx = etl.argsort(etl.minimum(fit_1, fit_2))[:elite_pop_size]
    fit_1_e = etl.gather(fit_1, elite_idx, axis=0)
    fit_2_e = etl.gather(fit_2, elite_idx, axis=0)
    fitness_elite = etl.concatenate([fit_1_e, fit_2_e], axis=0)
    sigma_fitness = etl.std(fitness_elite, ddof=1) + 1e-5
    fit_diff = fit_1_e - fit_2_e
    noise_1_e = etl.gather(state.noise[:half], elite_idx, axis=0)
    fit_diff_noise = etl.dot(enp.expand_dims(fit_diff, axis=0), noise_1_e)[0]
    theta_grad = 1.0 / (elite_pop_size * sigma_fitness) * fit_diff_noise
    if config.optimizer is None:
        center = state.center - config.lr * theta_grad
        exp_avg, exp_avg_sq = state.exp_avg, state.exp_avg_sq
    else:
        center, exp_avg, exp_avg_sq = adam_single_tensor(
            state.center,
            theta_grad,
            state.exp_avg,
            state.exp_avg_sq,
            0.9,
            0.999,
            config.lr,
        )
    best_fitness = etl.minimum(state.best_fitness, etl.min(fitness))
    return replace(
        state,
        center=center,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        best_fitness=best_fitness,
    )
