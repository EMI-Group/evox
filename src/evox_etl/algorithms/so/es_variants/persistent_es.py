"""Functional ETL port of the torch evox PersistentES algorithm (step
protocol).

Plain (non-`@etl.defn`) functions: they may only be called inside an active
trace (a function passed to `etl.build`/`etl.run`), since ETL has no eager
mode. Semantics mirror the torch original in
`src/evox/algorithms/so/es_variants/persistent_es.py` exactly: `step` owns the
WHOLE generation (mirrored perturbation sampling + accumulation →
`evaluate(candidates)` → persistent gradient update), with the
workflow-injected `evaluate` closure standing in for torch's
`self.evaluate(population)`.

Reference: Unbiased Gradient Estimation in Unrolled Computation Graphs with
Persistent Evolution Strategies
(http://proceedings.mlr.press/v139/vicol21a.html)
"""

from dataclasses import dataclass, replace
from typing import Callable, Literal, Optional

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from evox_etl.algorithms._config_utils import (
    ArrayLike,
    bake_float32_constant,
    require_gt,
    to_float_tuple,
)

from .adam_step import adam_single_tensor

Tensor = etl.SymbolicTensor


@dataclass(frozen=True)
class PersistentESConfig:
    """PersistentES hyperparameters (same names/defaults as the torch evox __init__).

    Dumb frozen config storing plain static leaves: `center_init` arrives as a
    flat float32 tuple, normalized by `make_persistent_es` (static config —
    baked into the graph as a constant at init).
    """

    pop_size: int
    center_init: tuple[float, ...]
    optimizer: Optional[Literal["adam"]] = None
    lr: float = 0.05
    sigma: float = 0.03
    T: int = 100  # inner problem length
    K: int = 10
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01


def make_persistent_es(
    pop_size: int,
    center_init: ArrayLike,
    optimizer: Optional[Literal["adam"]] = None,
    lr: float = 0.05,
    sigma: float = 0.03,
    T: int = 100,
    K: int = 10,
    sigma_decay: float = 1.0,
    sigma_limit: float = 0.01,
) -> PersistentESConfig:
    """Build a `PersistentESConfig`, normalizing `center_init` and validating fields."""
    require_gt("pop_size", pop_size, 1)
    if pop_size % 2 != 0:
        raise ValueError(f"pop_size must be even (mirrored sampling), got {pop_size!r}")
    return PersistentESConfig(
        pop_size=pop_size,
        center_init=to_float_tuple(center_init, dtype=np.float32),
        optimizer=optimizer,
        lr=lr,
        sigma=sigma,
        T=T,
        K=K,
        sigma_decay=sigma_decay,
        sigma_limit=sigma_limit,
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
    center = bake_float32_constant(config.center_init)
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


def step(
    config: PersistentESConfig,
    state: PersistentESState,
    evaluate: Callable[[Tensor], Tensor],
) -> PersistentESState:
    """Run ONE full generation: sample a mirrored population and accumulate
    the perturbations for the persistent gradient estimate, evaluate it, and
    update the center; the accumulator is reset when the inner problem wraps
    around.

    ``evaluate`` is the workflow-injected traced closure (opaque; minimization
    semantics).
    """
    dim = len(config.center_init)
    half = config.pop_size // 2
    f32 = np.dtype("float32")
    key, subkey = random.split(state.key)
    pos = random.normal(subkey, (half, dim), dtype=f32) * state.sigma
    perts = etl.concatenate([pos, -pos], axis=0)
    pert_accum = state.pert_accum + perts
    population = state.center + perts
    state = replace(state, pert_accum=pert_accum, key=key)

    fitness = evaluate(population)

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
    pert_accum = etl.select(reset, enp.zeros((config.pop_size, dim), dtype=f32), state.pert_accum)
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


def record_step(
    config: PersistentESConfig,
    state: PersistentESState,
    candidate: Tensor | None,
    fitness: Tensor | None,
) -> dict[str, Tensor]:
    """Host-side auxiliary-info hook for the monitor (mirrors torch
    `PersistentES.record_step`).

    PLAIN module-level function (NOT ``@etl.defn``): `StdWorkflow` calls it
    HOST-SIDE on CONCRETE etl tensors of the POST-step state, outside any traced
    graph. `candidate`/`fitness` are accepted-but-ignored, as in torch. Returns
    the distribution summary the monitor's `record_auxiliary` stores.
    """
    return {"center": state.center, "sigma": state.sigma}