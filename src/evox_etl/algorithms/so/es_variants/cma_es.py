"""Functional ETL port of the torch evox `cma_es.py` algorithm (CMA-ES).

Plain (non-`@etl.defn`) functions: `init`, `ask`, `tell`. They may only be
called inside an active trace (a function passed to `etl.build`/
`etl.evaluate`), since ETL has no eager mode. Semantics mirror the torch
original in `src/evox/algorithms/so/es_variants/cma_es.py` exactly; the
`CMAESConfig` fields match its `__init__` signature minus `device`.
"""

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

from .sort_utils import sort_by_key

Tensor = etl.SymbolicTensor


@dataclass(frozen=True)
class CMAESConfig:
    """CMA-ES hyperparameters, mirroring torch `CMAES.__init__` minus `device`.

    Array-like fields (`mean_init`, `weights`) accept numpy arrays or tuples
    and are normalized to flat float32 tuples in `__post_init__`.
    """

    mean_init: "np.ndarray | tuple[float, ...]"
    sigma: float
    pop_size: Optional[int] = None
    weights: "Optional[np.ndarray | tuple[float, ...]]" = None

    def __post_init__(self) -> None:
        assert self.sigma > 0, "sigma must be greater than 0."
        if self.pop_size is not None:
            assert self.pop_size > 0, "pop_size must be greater than 0."
        object.__setattr__(
            self,
            "mean_init",
            tuple(np.asarray(self.mean_init, dtype=np.float32).tolist()),
        )
        if self.weights is not None:
            object.__setattr__(
                self,
                "weights",
                tuple(np.asarray(self.weights, dtype=np.float32).tolist()),
            )


@dataclass(frozen=True)
class CMAESState:
    """CMA-ES mutable state; every leaf is an ETL tensor (no Python scalars)."""

    iteration: Tensor  # int32 scalar, incremented at the start of each step
    C: Tensor  # (dim, dim) covariance matrix
    C_invsqrt: Tensor  # (dim, dim)
    B: Tensor  # (dim, dim) eigenvector matrix, stored transposed
    D: Tensor  # (dim, dim)
    p_sigma: Tensor  # (dim,) evolution path for the step size
    p_c: Tensor  # (dim,) evolution path for the covariance matrix
    mean: Tensor  # (1, dim)
    sigma: Tensor  # scalar step size
    weights: Tensor  # (1, mu) recombination weights
    y: Tensor  # (pop_size, dim) sampled noise of the latest ask
    best_fitness: Tensor  # scalar, best fitness seen so far
    key: Tensor  # int64 scalar RNG key


@dataclass(frozen=True)
class _CMAESParams:
    """Static scalar parameters derived from the config (computed at trace time)."""

    dim: int
    pop_size: int
    mu: int
    mu_eff: float
    user_weights: Optional[tuple[float, ...]]
    chi_n: float
    c_sigma: float
    d_sigma: float
    c_c: float
    c_1: float
    c_mu: float
    decomp_per_iter: int


def _recombination_params(
    config: CMAESConfig, pop_size: int, mu: int
) -> tuple[float, Optional[tuple[float, ...]]]:
    """Return `(mu_eff, user_weights)` as Python scalars.

    `user_weights` is the user-provided tuple, or `None` when the default
    (closed-form) weights are used, mirroring torch's `float(self.mu_eff)`.
    """
    if config.weights is None:
        w = np.float32(math.log((pop_size + 1) / 2)) - np.log(
            np.arange(1, mu + 1, dtype=np.float32)
        )
        w = w / w.sum()
        return float(w.sum() ** 2 / (w**2).sum()), None
    w = np.asarray(config.weights, dtype=np.float32)
    return float(w.sum() ** 2 / (w**2).sum()), config.weights


def _derive(config: CMAESConfig) -> _CMAESParams:
    """Compute all static CMA-ES scalar parameters from the config (torch `__init__`)."""
    dim = len(config.mean_init)
    pop_size = (
        4 + math.floor(3 * math.log(dim))
        if config.pop_size is None
        else config.pop_size
    )
    mu = pop_size // 2
    mu_eff, user_weights = _recombination_params(config, pop_size, mu)
    chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
    d_sigma = 1 + 2 * max(math.sqrt((mu_eff - 1) / (dim + 1)) - 1, 0) + c_sigma
    # Hansen-canonical formula (the mu_eff+2 variant drives c_c above 2 for
    # large pop_size, making c_c * (2 - c_c) negative and sqrt(...) NaN).
    c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
    c_mu = min(
        1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((float(dim) + 2) ** 2 + mu_eff)
    )
    decomp_per_iter = max(math.floor(1 / (c_1 + c_mu) / dim / 10), 1)
    return _CMAESParams(
        dim, pop_size, mu, mu_eff, user_weights, chi_n, c_sigma, d_sigma, c_c,
        c_1, c_mu, decomp_per_iter,
    )


def _decompose(
    C: Tensor, B: Tensor, D: Tensor, C_invsqrt: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Eigendecompose the covariance matrix (true branch of the cond)."""
    C = (C + etl.transpose(C, (1, 0))) / 2
    D_evals, B_vecs = etl.eigh(C)
    D_evals = etl.maximum(D_evals, 1e-8)
    C_invsqrt = etl.dot(
        B_vecs,
        etl.dot(etl.diag(1.0 / etl.sqrt(D_evals)), etl.transpose(B_vecs, (1, 0))),
    )
    D_new = etl.dot(B_vecs, etl.diag(etl.sqrt(D_evals)))
    return etl.transpose(B_vecs, (1, 0)), D_new, C_invsqrt


def _no_decompose(
    C: Tensor, B: Tensor, D: Tensor, C_invsqrt: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Keep the previous decomposition (false branch of the cond)."""
    return B, D, C_invsqrt


def init(config: CMAESConfig, key: Tensor) -> CMAESState:
    """Create the initial CMA-ES state (draws nothing — key stored as-is)."""
    p = _derive(config)
    f32 = np.dtype("float32")
    mean_c = etl.ops.constant(
        etl.core.tensor(np.asarray(config.mean_init, dtype=np.float32))
    )
    mean = enp.expand_dims(mean_c, axis=0)  # (1, dim)
    sigma = enp.full((), config.sigma, dtype=f32)
    eye = etl.eye(p.dim)
    if config.weights is None:
        w = math.log((p.pop_size + 1) / 2) - etl.log(
            enp.arange(1, p.mu + 1, dtype=f32)
        )
        weights = enp.expand_dims(w / etl.sum(w), axis=0)  # (1, mu)
    else:
        w_c = etl.ops.constant(
            etl.core.tensor(np.asarray(config.weights, dtype=np.float32))
        )
        weights = etl.reshape(w_c, (1, p.mu))
    return CMAESState(
        iteration=enp.full((), 0, dtype=np.dtype("int32")),
        C=eye,
        C_invsqrt=eye,
        B=eye,
        D=eye,
        p_sigma=enp.zeros((p.dim,), dtype=f32),
        p_c=enp.zeros((p.dim,), dtype=f32),
        mean=mean,
        sigma=sigma,
        weights=weights,
        y=enp.zeros((p.pop_size, p.dim), dtype=f32),
        best_fitness=enp.full((), np.inf, dtype=f32),
        key=key,
    )


def ask(config: CMAESConfig, state: CMAESState) -> tuple[Tensor, CMAESState]:
    """Sample the population and increment the iteration counter."""
    p = _derive(config)
    f32 = np.dtype("float32")
    iteration = etl.cast(state.iteration + 1, np.dtype("int32"))
    key, subkey = random.split(state.key)
    noise = random.normal(subkey, (p.pop_size, p.dim), dtype=f32)
    y = etl.dot(noise, state.D)
    y = etl.dot(y, state.B)
    population = state.mean + state.sigma * y
    return population, replace(state, iteration=iteration, y=y, key=key)


def tell(config: CMAESConfig, state: CMAESState, fitness: Tensor) -> CMAESState:
    """Update mean, covariance matrix and step size from the evaluated population."""
    p = _derive(config)
    f32 = np.dtype("float32")
    # ask stored the sampled noise in state.y; mean/sigma are still the
    # pre-update values, so this reconstructs the population ask returned.
    population = state.mean + state.sigma * state.y
    fitness, population = sort_by_key(fitness, population)
    population_selected = population[: p.mu]
    old_mean = state.mean
    new_mean = old_mean + etl.dot(state.weights, population_selected - old_mean)
    delta_mean = (new_mean - old_mean)[0]

    p_sigma = (1 - p.c_sigma) * state.p_sigma + math.sqrt(
        p.c_sigma * (2 - p.c_sigma) * p.mu_eff
    ) * etl.dot(state.C_invsqrt, enp.expand_dims(delta_mean, axis=1))[
        :, 0
    ] / state.sigma

    exponent = etl.cast(2 * state.iteration, f32)
    norm_term = etl.norm(p_sigma) / etl.sqrt(
        1 - etl.exp(exponent * math.log(1 - p.c_sigma))
    )
    h_sigma = etl.cast(
        norm_term < (1.4 + 2 / (p.dim + 1)) * p.chi_n, f32
    )

    p_c = (1 - p.c_c) * state.p_c + h_sigma * math.sqrt(
        p.c_c * (2 - p.c_c) * p.mu_eff
    ) * delta_mean / state.sigma

    y_sel = (population_selected - old_mean) / state.sigma
    # torch parity: `p_c @ p_c.T` on 1-D p_c is a SCALAR dot product in torch
    # (1-D .T is a no-op), broadcasting additively into every element of C —
    # an isotropic c_1-scaled inflation, NOT the canonical rank-one outer
    # product. See "Design Decisions" in this directory's CONTEXT.md.
    pc_norm_sq = etl.sum(p_c * p_c)  # 0-d; etl.dot needs rank >= 2
    C_new = (
        (1 - p.c_1 - p.c_mu) * state.C
        + p.c_1 * (pc_norm_sq + (1 - h_sigma) * p.c_c * (2 - p.c_c) * state.C)
        + p.c_mu * etl.dot(etl.transpose(y_sel, (1, 0)) * state.weights, y_sel)
    )
    sigma_new = state.sigma * etl.exp(
        p.c_sigma / p.d_sigma * (etl.norm(p_sigma) / p.chi_n - 1)
    )

    pred = etl.equal(etl.remainder(state.iteration, p.decomp_per_iter), 0)
    B_new, D_new, C_invsqrt_new = etl.cond(
        pred, _decompose, _no_decompose, C_new, state.B, state.D, state.C_invsqrt
    )

    return CMAESState(
        iteration=state.iteration,
        C=C_new,
        C_invsqrt=C_invsqrt_new,
        B=B_new,
        D=D_new,
        p_sigma=p_sigma,
        p_c=p_c,
        mean=new_mean,
        sigma=sigma_new,
        weights=state.weights,
        y=state.y,
        best_fitness=etl.minimum(state.best_fitness, etl.min(fitness)),
        key=state.key,
    )
