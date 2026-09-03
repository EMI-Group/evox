"""Functional ETL port of the torch evox ASEBO algorithm (plain functions).

Reference (read-only): ``src/evox/algorithms/so/es_variants/asebo.py``
(Adaptive ES-Active Subspaces for Blackbox Optimization).  ETL has no eager
mode, so ``init``/``ask``/``tell`` are plain functions traced via
``etl.build``/``etl.run``; the torch ``step`` is split at ``self.evaluate``.
``lr_decay``/``lr_limit`` are kept for API parity but unused (as in torch).
"""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from etl.core import SymbolicTensor

from .adam_step import adam_single_tensor

F32 = np.dtype("float32")


@dataclass(frozen=True)
class ASEBOConfig:
    """Hyperparameters of the ASEBO algorithm (mirrors the torch ``__init__``)."""

    pop_size: int
    center_init: np.ndarray
    optimizer: Literal["adam"] | None = None
    lr: float = 0.05
    lr_decay: float = 1.0
    lr_limit: float = 0.001
    sigma: float = 0.03
    sigma_decay: float = 1.0
    sigma_limit: float = 0.01
    subspace_dims: int | None = None

    def __post_init__(self):
        assert self.pop_size > 1, f"pop_size must be > 1, got {self.pop_size}"
        assert self.optimizer in (None, "adam"), (
            f"unsupported optimizer {self.optimizer!r}; only None or 'adam'"
        )
        if isinstance(self.center_init, np.ndarray):
            object.__setattr__(
                self,
                "center_init",
                tuple(np.asarray(self.center_init, dtype=np.float32).tolist()),
            )
        if self.subspace_dims is None:
            object.__setattr__(self, "subspace_dims", len(self.center_init))


@dataclass(frozen=True)
class ASEBOState:
    """Mutable state of ASEBO; all leaves are etl tensors."""

    center: SymbolicTensor
    grad_subspace: SymbolicTensor
    UUT: SymbolicTensor
    UUT_ort: SymbolicTensor
    sigma: SymbolicTensor
    alpha: SymbolicTensor
    gen_counter: SymbolicTensor
    z: SymbolicTensor
    exp_avg: SymbolicTensor
    exp_avg_sq: SymbolicTensor
    best_fitness: SymbolicTensor
    key: SymbolicTensor


def init(config: ASEBOConfig, key: SymbolicTensor) -> ASEBOState:
    """Create the initial state; ``center`` is baked as a graph constant."""
    dim = len(config.center_init)
    center = etl.ops.constant(
        etl.core.tensor(np.asarray(config.center_init, dtype=F32))
    )
    grad_subspace = enp.zeros((config.subspace_dims, dim), dtype=F32)
    UUT = enp.zeros((dim, dim), dtype=F32)
    UUT_ort = enp.zeros((dim, dim), dtype=F32)
    sigma = enp.full((), config.sigma, dtype=F32)
    alpha = enp.full((), 0.1, dtype=F32)
    gen_counter = enp.full((), 0.0, dtype=F32)
    z = enp.zeros((config.pop_size, dim), dtype=F32)
    exp_avg = enp.zeros((dim,), dtype=F32)
    exp_avg_sq = enp.zeros((dim,), dtype=F32)
    best_fitness = enp.full((), np.inf, dtype=F32)
    return ASEBOState(
        center,
        grad_subspace,
        UUT,
        UUT_ort,
        sigma,
        alpha,
        gen_counter,
        z,
        exp_avg,
        exp_avg_sq,
        best_fitness,
        key,
    )


def ask(
    config: ASEBOConfig, state: ASEBOState
) -> tuple[SymbolicTensor, ASEBOState]:
    """Sample the mirrored population from the active-subspace covariance."""
    dim = len(config.center_init)
    half = config.pop_size // 2
    sub = config.subspace_dims

    key, subkey = random.split(state.key)

    # Active subspace via SVD of the gradient history.
    X = state.grad_subspace - etl.mean(state.grad_subspace, axes=0)
    U, _S, Vh = etl.svd(X)  # reduced: U (sub, k), Vh (k, dim), k = min(sub, dim)
    k = min(sub, dim)
    max_abs_cols = etl.argmax(etl.abs(U), axis=0)
    offset_idx = max_abs_cols + etl.cast(
        enp.arange(k, dtype=np.dtype("int64")), np.dtype("int64")
    ) * sub
    row_collected = etl.gather(enp.reshape(U, (-1,)), offset_idx, axis=0)
    signs = etl.sign(row_collected)
    U = U * signs
    Vh = Vh * enp.expand_dims(signs, axis=1)

    U2 = Vh[:half]
    UUT = etl.dot(etl.transpose(U2), U2)
    U_ort = Vh[half:]
    UUT_ort = etl.dot(etl.transpose(U_ort), U_ort)
    UUT = etl.select(
        state.gen_counter > sub, UUT, enp.zeros((dim, dim), dtype=F32)
    )

    cov = (
        state.sigma * (state.alpha / dim) * etl.eye(dim)
        + ((1.0 - state.alpha) / half) * UUT
    )
    chol = etl.cholesky(cov)
    noise = random.normal(subkey, (dim, half), mean=0.0, std=1.0, dtype=F32)
    z_plus = etl.transpose(etl.dot(chol, noise), (1, 0))
    z_plus = z_plus / etl.norm(z_plus, axis=1, keepdims=True)
    z = etl.concatenate([z_plus, -1.0 * z_plus], axis=0)
    population = state.center + z
    gen_counter = state.gen_counter + 1.0
    return population, replace(
        state, z=z, UUT=UUT, UUT_ort=UUT_ort, gen_counter=gen_counter, key=key
    )


def tell(
    config: ASEBOConfig, state: ASEBOState, fitness: SymbolicTensor
) -> ASEBOState:
    """Update center, alpha and sigma from the mirrored fitness pairs."""
    dim = len(config.center_init)
    half = config.pop_size // 2
    noise_1 = state.z[:half] / state.sigma
    fit_diff_noise = etl.dot(
        enp.expand_dims(fitness[:half] - fitness[half:], axis=0), noise_1
    )[0]
    theta_grad = 0.5 * fit_diff_noise
    alpha = etl.norm(
        etl.dot(enp.expand_dims(theta_grad, axis=0), state.UUT_ort)[0]
    ) / etl.norm(etl.dot(enp.expand_dims(theta_grad, axis=0), state.UUT)[0])
    alpha = etl.select(state.gen_counter > config.subspace_dims, alpha, 1.0)
    grad_subspace = etl.concatenate(
        [state.grad_subspace, enp.expand_dims(theta_grad, axis=0)], axis=0
    )[1:, :]
    theta_grad = theta_grad / (etl.norm(theta_grad) / dim + 1e-8)
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
        alpha=alpha,
        grad_subspace=grad_subspace,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        best_fitness=best_fitness,
    )
