"""Functional ETL port of the torch evox PersistentES algorithm.

Plain (non-`@etl.defn`) functions: they may only be called inside an active
trace (a function passed to `etl.build`/`etl.evaluate`), since ETL has no
eager mode. Semantics mirror the torch original in
`src/evox/algorithms/so/es_variants/persistent_es.py` exactly.

Reference: Unbiased Gradient Estimation in Unrolled Computation Graphs with
Persistent Evolution Strategies
(http://proceedings.mlr.press/v139/vicol21a.html)
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
class PersistentESConfig:
    """PersistentES hyperparameters (same names/defaults as the torch evox __init__).

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
        assert self.pop_size > 1 and self.pop_size % 2 == 0  # Population size must be even
        if isinstance(self.center_init, np.ndarray):
            object.__setattr__(
                self,
                "center_init",
                tuple(np.asarray(self.center_init, dtype=np.float32).tolist()),
            )


@dataclass(frozen=True)
class PersistentESState:
    """Mutable PersistentES state; all leaves are ETL tensors."""

    center: Tensor
    sigma: Tensor
    inner_step_counter: Tensor
    pert_accum: Tensor
    exp_avg: Tensor
    exp_avg_sq: Tensor
    best_fitness: Tensor
    key: Tensor


def init(config: PersistentESConfig, key: Tensor) -> PersistentESState:
    """Build the initial PersistentES state."""
    dim = len(config.center_init)
    f32 = np.dtype("float32")
    center = etl.ops.constant(etl.core.tensor(np.asarray(config.center_init, dtype=f32)))
    sigma = enp.full((), config.sigma, dtype=f32)
    inner_step_counter = enp.full((), 0.0, dtype=f32)
    pert_accum = enp.zeros((config.pop_size, dim), dtype=f32)
    exp_avg = enp.zeros((dim,), dtype=f32)
    exp_avg_sq = enp.zeros((dim,), dtype=f32)
    best_fitness = enp.full((), np.inf, dtype=f32)
    return PersistentESState(
        center=center,
        sigma=sigma,
        inner_step_counter=inner_step_counter,
        pert_accum=pert_accum,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        best_fitness=best_fitness,
        key=key,
    )


def ask(config: PersistentESConfig, state: PersistentESState) -> tuple[Tensor, PersistentESState]:
    """Sample a mirrored population; accumulate the perturbations for the
    persistent gradient estimate."""
    dim = len(config.center_init)
    half = config.pop_size // 2
    f32 = np.dtype("float32")
    key, subkey = random.split(state.key)
    pos = random.normal(subkey, (half, dim), dtype=f32) * state.sigma
    perts = etl.concatenate([pos, -pos], axis=0)
    pert_accum = state.pert_accum + perts
    population = state.center + perts
    state = replace(state, pert_accum=pert_accum, key=key)
    return population, state


def tell(config: PersistentESConfig, state: PersistentESState, fitness: Tensor) -> PersistentESState:
    """Update center from the persistent gradient estimate; reset the
    accumulator when the inner problem wraps around."""
    dim = len(config.center_init)
    pop_size = config.pop_size
    f32 = np.dtype("float32")
    theta_grad = etl.mean(
        state.pert_accum * enp.expand_dims(fitness, axis=1) / (state.sigma * state.sigma),
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
    inner_step_counter = state.inner_step_counter + config.K
    reset = inner_step_counter >= config.T
    inner_step_counter = etl.select(reset, 0, inner_step_counter)
    pert_accum = etl.select(reset, enp.zeros((pop_size, dim), dtype=f32), state.pert_accum)
    sigma = etl.maximum(config.sigma_decay * state.sigma, config.sigma_limit)
    best_fitness = etl.minimum(state.best_fitness, etl.min(fitness))
    return replace(
        state,
        center=center,
        inner_step_counter=inner_step_counter,
        pert_accum=pert_accum,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        sigma=sigma,
        best_fitness=best_fitness,
    )
