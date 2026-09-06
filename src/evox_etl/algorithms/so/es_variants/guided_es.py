"""Functional ETL port of the torch evox GuidedES algorithm.

Plain (non-`@etl.defn`) functions: they may only be called inside an active
trace (a function passed to `etl.build`/`etl.evaluate`), since ETL has no
eager mode. Semantics mirror the torch original in
`src/evox/algorithms/so/es_variants/guided_es.py` exactly.

Reference: Guided evolutionary strategies (https://arxiv.org/abs/1806.10230)
"""

from dataclasses import dataclass, replace
from typing import Literal, Optional

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
class GuidedESConfig:
    """GuidedES hyperparameters (same names/defaults as the torch evox __init__).

    Dumb frozen config storing plain static leaves: `center_init` arrives as a
    flat float32 tuple, normalized by `make_guided_es` (static config — baked
    into the graph as a constant at init).
    """

    pop_size: int
    center_init: tuple[float, ...]
    subspace_dims: Optional[int] = None
    optimizer: Optional[Literal["adam"]] = None
    sigma: float = 0.03
    lr: float = 60
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01


def make_guided_es(
    pop_size: int,
    center_init: ArrayLike,
    subspace_dims: Optional[int] = None,
    optimizer: Optional[Literal["adam"]] = None,
    sigma: float = 0.03,
    lr: float = 60,
    sigma_decay: float = 1.0,
    sigma_limit: float = 0.01,
) -> GuidedESConfig:
    """Build a `GuidedESConfig`, normalizing `center_init` and validating fields.

    `subspace_dims` is NOT derived here (it defaults to `dim` at trace time in
    init/ask when left None, mirroring the torch reference).
    """
    require_gt("pop_size", pop_size, 1)
    if pop_size % 2 != 0:
        raise ValueError(f"pop_size must be even (mirrored sampling), got {pop_size!r}")
    return GuidedESConfig(
        pop_size=pop_size,
        center_init=to_float_tuple(center_init, dtype=np.float32),
        subspace_dims=subspace_dims,
        optimizer=optimizer,
        sigma=sigma,
        lr=lr,
        sigma_decay=sigma_decay,
        sigma_limit=sigma_limit,
    )


@dataclass(frozen=True)
class GuidedESState:
    """Mutable GuidedES state; all leaves are ETL tensors."""

    center: Tensor
    alpha: Tensor
    sigma: Tensor
    grad_subspace: Tensor
    z: Tensor
    exp_avg: Tensor
    exp_avg_sq: Tensor
    best_fitness: Tensor
    key: Tensor


def init(config: GuidedESConfig, key: Tensor) -> GuidedESState:
    """Draw the initial state (including the random grad_subspace)."""
    dim = len(config.center_init)
    pop_size = config.pop_size
    subspace_dims = dim if config.subspace_dims is None else config.subspace_dims
    f32 = np.dtype("float32")
    center = bake_float32_constant(config.center_init)
    alpha = enp.full((), 0.5, dtype=f32)
    sigma = enp.full((), config.sigma, dtype=f32)
    key, subkey = random.split(key)
    grad_subspace = random.normal(subkey, (subspace_dims, dim), dtype=f32)
    z = enp.zeros((pop_size, dim), dtype=f32)
    exp_avg = enp.zeros((dim,), dtype=f32)
    exp_avg_sq = enp.zeros((dim,), dtype=f32)
    best_fitness = enp.full((), np.inf, dtype=f32)
    return GuidedESState(
        center=center,
        alpha=alpha,
        sigma=sigma,
        grad_subspace=grad_subspace,
        z=z,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        best_fitness=best_fitness,
        key=key,
    )


def ask(config: GuidedESConfig, state: GuidedESState) -> tuple[Tensor, GuidedESState]:
    """Sample a mirrored population around the current center."""
    dim = len(config.center_init)
    pop_size = config.pop_size
    subspace_dims = dim if config.subspace_dims is None else config.subspace_dims
    half = pop_size // 2
    f32 = np.dtype("float32")
    key, subkey = random.split(state.key)
    key, subkey2 = random.split(key)
    a = state.sigma * etl.sqrt(state.alpha / dim)
    c = state.sigma * etl.sqrt((1.0 - state.alpha) / subspace_dims)
    eps_full = random.normal(subkey, (dim, half), dtype=f32)
    eps_subspace = random.normal(subkey2, (subspace_dims, half), dtype=f32)
    q, _ = etl.qr(state.grad_subspace)
    z_plus = a * eps_full + c * etl.dot(q, eps_subspace)
    z_plus = etl.transpose(z_plus, (1, 0))
    z = etl.concatenate([z_plus, -z_plus], axis=0)
    population = state.center + z
    state = replace(state, z=z, key=key)
    return population, state


def tell(config: GuidedESConfig, state: GuidedESState, fitness: Tensor) -> GuidedESState:
    """Update center, grad_subspace and sigma from the evaluated population."""
    pop_size = config.pop_size
    half = pop_size // 2
    noise = state.z / state.sigma
    noise_1 = noise[:half]
    fit_diff = fitness[:half] - fitness[half:]
    fit_diff_noise = etl.dot(enp.expand_dims(fit_diff, axis=0), noise_1)[0]
    theta_grad = (1.0 / pop_size) * fit_diff_noise
    grad_subspace = etl.concatenate(
        [state.grad_subspace, enp.expand_dims(theta_grad, axis=0)], axis=0
    )[1:, :]
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
    sigma = etl.maximum(config.sigma_decay * state.sigma, config.sigma_limit)
    best_fitness = etl.minimum(state.best_fitness, etl.min(fitness))
    return replace(
        state,
        center=center,
        grad_subspace=grad_subspace,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        sigma=sigma,
        best_fitness=best_fitness,
    )
