"""Functional ETL port of the torch evox OpenES algorithm.

Plain (non-`@etl.defn`) functions: they may only be called inside an active
trace (a function passed to `etl.build`/`etl.run`), since ETL has no eager
mode.  Semantics mirror the torch original in
`src/evox/algorithms/so/es_variants/open_es.py` exactly.

OpenES is described in "Evolution Strategies as a Scalable Alternative to
Reinforcement Learning" (https://arxiv.org/abs/1703.03864).
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
class OpenESConfig:
    """Frozen hyperparameters (torch `OpenES.__init__` minus `device`).

    ``center_init`` may be passed as an ``np.ndarray``; it is normalized to a
    flat tuple of float32 values (hashable static config arg).
    """

    pop_size: int
    center_init: tuple[float, ...]
    learning_rate: float
    noise_stdev: float
    optimizer: Literal["adam"] | None = None
    mirrored_sampling: bool = True

    def __post_init__(self) -> None:
        assert self.noise_stdev > 0, "noise_stdev must be greater than 0."
        assert self.learning_rate > 0, "learning_rate must be greater than 0."
        assert self.pop_size > 0, "pop_size must be greater than 0."
        if self.mirrored_sampling:
            assert self.pop_size % 2 == 0, (
                "When mirrored_sampling is True, pop_size must be a multiple of 2."
            )
        assert self.optimizer in [None, "adam"], "optimizer must be None or 'adam'."
        if isinstance(self.center_init, np.ndarray):
            object.__setattr__(
                self,
                "center_init",
                tuple(np.asarray(self.center_init, dtype=np.float32).tolist()),
            )


@dataclass(frozen=True)
class OpenESState:
    """Algorithm state: tensor leaves only."""

    center: Tensor  # (dim,) current search center
    noise: Tensor  # (pop_size, dim) noise sampled by the last ask
    exp_avg: Tensor  # (dim,) Adam first moment (always carried)
    exp_avg_sq: Tensor  # (dim,) Adam second moment (always carried)
    best_fitness: Tensor  # () best fitness seen so far
    key: Tensor  # () RNG key


def init(config: OpenESConfig, key: Tensor) -> OpenESState:
    """Create the initial state; no randomness is drawn (key passed through)."""
    dim = len(config.center_init)
    center = etl.ops.constant(
        etl.core.tensor(np.asarray(config.center_init, dtype=np.float32))
    )
    noise = enp.zeros((config.pop_size, dim), dtype=F32)
    exp_avg = enp.zeros((dim,), dtype=F32)
    exp_avg_sq = enp.zeros((dim,), dtype=F32)
    best_fitness = enp.full((), np.inf, dtype=F32)
    return OpenESState(
        center=center,
        noise=noise,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        best_fitness=best_fitness,
        key=key,
    )


def ask(config: OpenESConfig, state: OpenESState) -> tuple[Tensor, OpenESState]:
    """Sample the population around the center (mirrored or plain Gaussian noise)."""
    dim = len(config.center_init)
    key, subkey = random.split(state.key)
    if config.mirrored_sampling:
        noise = random.normal(subkey, (config.pop_size // 2, dim), dtype=F32)
        noise = etl.concatenate([noise, -noise], axis=0)
    else:
        noise = random.normal(subkey, (config.pop_size, dim), dtype=F32)
    population = enp.expand_dims(state.center, axis=0) + config.noise_stdev * noise
    return population, replace(state, noise=noise, key=key)


def tell(config: OpenESConfig, state: OpenESState, fitness: Tensor) -> OpenESState:
    """Update the center from the fitness via the ES gradient estimate."""
    grad = (
        etl.dot(enp.expand_dims(fitness, axis=0), state.noise)[0]
        / config.pop_size
        / config.noise_stdev
    )
    if config.optimizer is None:
        center = state.center - config.learning_rate * grad
        exp_avg, exp_avg_sq = state.exp_avg, state.exp_avg_sq
    else:
        center, exp_avg, exp_avg_sq = adam_single_tensor(
            state.center,
            grad,
            state.exp_avg,
            state.exp_avg_sq,
            0.9,
            0.999,
            config.learning_rate,
        )
    best_fitness = etl.minimum(state.best_fitness, etl.min(fitness))
    return replace(
        state,
        center=center,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        best_fitness=best_fitness,
    )
