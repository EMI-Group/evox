"""Functional ETL port of the torch evox NoiseReuseES algorithm.

Plain (non-`@etl.defn`) functions: they may only be called inside an active
trace (a function passed to `etl.build`/`etl.evaluate`), since ETL has no
eager mode. Semantics mirror the torch original in
`src/evox/algorithms/so/es_variants/noise_reuse_es.py` exactly.

Reference: Noise-Reuse in Online Evolution Strategies
(https://arxiv.org/pdf/2304.12180.pdf)
"""

from dataclasses import dataclass, replace
from typing import Literal, Optional, Union

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from .adam_step import adam_single_tensor

Tensor = etl.SymbolicTensor


@dataclass(frozen=True)
class NoiseReuseESConfig:
    """NoiseReuseES hyperparameters (same names/defaults as the torch evox __init__).

    `center_init` may be passed as a numpy array; it is normalized to a flat
    tuple of float32 values (static config — baked into the graph).
    """

    pop_size: int
    center_init: Union[np.ndarray, tuple]
    optimizer: Optional[Literal["adam"]] = None
    lr: float = 0.05
    sigma: float = 0.03
    T: int = 100  # inner problem length
    K: int = 10
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01

    def __post_init__(self) -> None:
        assert self.pop_size > 1
        if isinstance(self.center_init, np.ndarray):
            object.__setattr__(
                self,
                "center_init",
                tuple(np.asarray(self.center_init, dtype=np.float32).tolist()),
            )


@dataclass(frozen=True)
class NoiseReuseESState:
    """Mutable NoiseReuseES state; all leaves are ETL tensors."""

    center: Tensor
    sigma: Tensor
    inner_step_counter: Tensor
    unroll_pert: Tensor
    exp_avg: Tensor
    exp_avg_sq: Tensor
    best_fitness: Tensor
    key: Tensor


def init(config: NoiseReuseESConfig, key: Tensor) -> NoiseReuseESState:
    """Build the initial NoiseReuseES state."""
    dim = len(config.center_init)
    f32 = np.dtype("float32")
    center = etl.ops.constant(etl.core.tensor(np.asarray(config.center_init, dtype=f32)))
    sigma = enp.full((), config.sigma, dtype=f32)
    inner_step_counter = enp.full((), 0.0, dtype=f32)
    unroll_pert = enp.zeros((config.pop_size, dim), dtype=f32)
    exp_avg = enp.zeros((dim,), dtype=f32)
    exp_avg_sq = enp.zeros((dim,), dtype=f32)
    best_fitness = enp.full((), np.inf, dtype=f32)
    return NoiseReuseESState(
        center=center,
        sigma=sigma,
        inner_step_counter=inner_step_counter,
        unroll_pert=unroll_pert,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        best_fitness=best_fitness,
        key=key,
    )


def ask(config: NoiseReuseESConfig, state: NoiseReuseESState) -> tuple[Tensor, NoiseReuseESState]:
    """Sample a mirrored population, reusing the previous perturbations when the
    inner-problem counter has not wrapped around."""
    dim = len(config.center_init)
    half = config.pop_size // 2
    f32 = np.dtype("float32")
    key, subkey = random.split(state.key)
    pos = random.normal(subkey, (half, dim), dtype=f32) * state.sigma
    perturbations = etl.concatenate([pos, -pos], axis=0)
    unroll_pert = etl.select(
        etl.equal(state.inner_step_counter, 0), perturbations, state.unroll_pert
    )
    population = state.center + unroll_pert
    state = replace(state, unroll_pert=unroll_pert, key=key)
    return population, state


def tell(config: NoiseReuseESConfig, state: NoiseReuseESState, fitness: Tensor) -> NoiseReuseESState:
    """Update center, inner-step counter and sigma from the evaluated population."""
    theta_grad = etl.mean(
        state.unroll_pert * enp.expand_dims(fitness, axis=1) / (state.sigma * state.sigma),
        axes=0,
    )
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
    inner_step_counter = etl.select(
        state.inner_step_counter + config.K >= config.T,
        0,
        state.inner_step_counter + config.K,
    )
    sigma = etl.maximum(config.sigma_decay * state.sigma, config.sigma_limit)
    best_fitness = etl.minimum(state.best_fitness, etl.min(fitness))
    return replace(
        state,
        center=center,
        inner_step_counter=inner_step_counter,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        sigma=sigma,
        best_fitness=best_fitness,
    )
