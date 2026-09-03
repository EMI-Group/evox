"""Functional ETL port of the torch evox xNES / SeparableNES algorithms.

Plain-function (non-``@etl.defn``) ``init``/``ask``/``tell`` split of the torch
classes in ``src/evox/algorithms/so/es_variants/nes.py`` (read-only reference).
ETL has no eager mode: every op runs inside functions invoked via
``etl.build``/``etl.run``.  The shared module-level ``init``/``ask``/``tell``
dispatch on the (static) config type, so ``run_generations`` can drive either
algorithm through one module.

Deviation from torch: the torch ``__init__`` default pop_size computes
``4 + math.floor(3 * math.log(self.dim))`` while ``self.dim`` is still
undefined at that point (AttributeError); the port uses the local ``dim``.

References:
- Exponential Natural Evolution Strategies (https://dl.acm.org/doi/abs/10.1145/1830483.1830557)
- Natural Evolution Strategies (https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf)
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random
from etl.core import SymbolicTensor

F32 = np.dtype("float32")


def _constant_1d(values: Any) -> SymbolicTensor:
    """Bake a flat config field as an f32 graph constant."""
    return etl.ops.constant(etl.core.tensor(np.asarray(values, dtype=F32)))


def _default_recombination_weights(pop_size: int) -> SymbolicTensor:
    """log-utility weights: clip(log(pop/2+1) - log(i+1), 0), zero-summed."""
    w = etl.clamp(
        math.log(pop_size / 2 + 1) - etl.log(enp.arange(1, pop_size + 1, dtype=F32)),
        0.0,
        np.inf,
    )
    return w / etl.sum(w) - 1.0 / pop_size


@dataclass(frozen=True)
class XNESConfig:
    """xNES hyperparameters (mirrors the torch ``XNES.__init__`` minus ``device``).

    Array fields are normalized to flat tuples of f32 Python floats in
    ``__post_init__`` (ndarray leaves are rejected as static etl.build args).
    """

    init_mean: Any
    init_covar: Any
    pop_size: int | None = None
    recombination_weights: Any | None = None
    learning_rate_mean: float | None = None
    learning_rate_var: float | None = None
    learning_rate_B: float | None = None
    covar_as_cholesky: bool = False

    def __post_init__(self) -> None:
        # NOTE: torch uses self.dim here while it is still undefined (bug) -> local dim.
        dim = np.asarray(self.init_mean).shape[0]
        if self.pop_size is None:
            object.__setattr__(self, "pop_size", 4 + math.floor(3 * math.log(dim)))
        assert self.pop_size > 0
        if self.learning_rate_mean is None:
            object.__setattr__(self, "learning_rate_mean", 1)
        if self.learning_rate_var is None:
            object.__setattr__(
                self,
                "learning_rate_var",
                (9 + 3 * math.log(dim)) / 5 / math.pow(dim, 1.5),
            )
        if self.learning_rate_B is None:
            object.__setattr__(self, "learning_rate_B", self.learning_rate_var)
        assert (
            self.learning_rate_mean > 0
            and self.learning_rate_var > 0
            and self.learning_rate_B > 0
        )
        if self.recombination_weights is not None:
            w = np.asarray(self.recombination_weights)
            assert (w[1:] <= w[:-1]).all(), (
                "recombination_weights must be in descending order"
            )
            object.__setattr__(self, "recombination_weights", tuple(w.astype(F32).tolist()))
        object.__setattr__(self, "init_mean", tuple(np.asarray(self.init_mean, dtype=F32).tolist()))
        object.__setattr__(
            self,
            "init_covar",
            tuple(tuple(row) for row in np.asarray(self.init_covar, dtype=F32).tolist()),
        )


@dataclass(frozen=True)
class XNESState:
    """xNES state — tensor leaves only, ``key`` last."""

    mean: SymbolicTensor
    sigma: SymbolicTensor
    B: SymbolicTensor
    recombination_weights: SymbolicTensor
    noise: SymbolicTensor
    best_fitness: SymbolicTensor
    key: SymbolicTensor


def _xn_init(config: XNESConfig, key: SymbolicTensor) -> XNESState:
    """Initial state: mean, sigma (geometric mean of the covar diagonal) and B."""
    dim = len(config.init_mean)
    pop_size = config.pop_size
    mean = _constant_1d(config.init_mean)
    covar = etl.ops.constant(
        etl.core.tensor(np.asarray(config.init_covar, dtype=F32))
    )
    if not config.covar_as_cholesky:
        covar = etl.cholesky(covar)
    if config.recombination_weights is None:
        weights = _default_recombination_weights(pop_size)
    else:
        weights = _constant_1d(config.recombination_weights)
    sigma = enp.power(etl.prod(etl.diagonal(covar)), 1.0 / dim)
    B = covar / sigma
    return XNESState(
        mean=mean,
        sigma=sigma,
        B=B,
        recombination_weights=weights,
        noise=enp.zeros((pop_size, dim), dtype=F32),
        best_fitness=enp.full((), np.inf, dtype=F32),
        key=key,
    )


def _xn_ask(config: XNESConfig, state: XNESState) -> tuple[SymbolicTensor, XNESState]:
    """Sample the population around the mean with the current covariance factor B."""
    pop_size = config.pop_size
    dim = len(config.init_mean)
    key, subkey = random.split(state.key)
    noise = random.normal(subkey, (pop_size, dim), mean=0.0, std=1.0, dtype=F32)
    population = state.mean + state.sigma * etl.dot(noise, etl.transpose(state.B, (1, 0)))
    return population, XNESState(
        mean=state.mean,
        sigma=state.sigma,
        B=state.B,
        recombination_weights=state.recombination_weights,
        noise=noise,
        best_fitness=state.best_fitness,
        key=key,
    )


def _xn_tell(config: XNESConfig, state: XNESState, fitness: SymbolicTensor) -> XNESState:
    """Natural-gradient update of mean, sigma and B from the ranked samples."""
    dim = len(config.init_mean)
    order = etl.argsort(fitness)
    noise = etl.gather(state.noise, order, axis=0)
    weights = state.recombination_weights
    Ind = etl.eye(dim)
    grad_delta = etl.sum(enp.expand_dims(weights, axis=1) * noise, axes=0)
    grad_M = etl.dot(etl.transpose(noise, (1, 0)) * weights, noise) - etl.sum(weights) * Ind
    grad_sigma = etl.sum(etl.diagonal(grad_M)) / dim
    grad_B = grad_M - grad_sigma * Ind
    mean = (
        state.mean
        + config.learning_rate_mean
        * state.sigma
        * etl.dot(state.B, enp.expand_dims(grad_delta, axis=1))[:, 0]
    )
    sigma = state.sigma * etl.exp(config.learning_rate_var / 2 * grad_sigma)
    B = etl.dot(state.B, etl.matrix_exp(config.learning_rate_B / 2 * grad_B))
    best_fitness = etl.minimum(state.best_fitness, etl.min(fitness))
    return XNESState(
        mean=mean,
        sigma=sigma,
        B=B,
        recombination_weights=state.recombination_weights,
        noise=state.noise,
        best_fitness=best_fitness,
        key=state.key,
    )


@dataclass(frozen=True)
class SeparableNESConfig:
    """SeparableNES hyperparameters (mirrors the torch ``__init__`` minus ``device``)."""

    init_mean: Any
    init_std: Any
    pop_size: int | None = None
    recombination_weights: Any | None = None
    learning_rate_mean: float | None = None
    learning_rate_var: float | None = None

    def __post_init__(self) -> None:
        # NOTE: torch uses self.dim here while it is still undefined (bug) -> local dim.
        dim = np.asarray(self.init_mean).shape[0]
        assert np.asarray(self.init_std).shape == (dim,)
        if self.pop_size is None:
            object.__setattr__(self, "pop_size", 4 + math.floor(3 * math.log(dim)))
        assert self.pop_size > 0
        if self.learning_rate_mean is None:
            object.__setattr__(self, "learning_rate_mean", 1)
        if self.learning_rate_var is None:
            object.__setattr__(
                self,
                "learning_rate_var",
                (3 + math.log(dim)) / 5 / math.sqrt(dim),
            )
        assert self.learning_rate_mean > 0 and self.learning_rate_var > 0
        if self.recombination_weights is not None:
            w = np.asarray(self.recombination_weights)
            assert w.shape == (self.pop_size,)
            object.__setattr__(self, "recombination_weights", tuple(w.astype(F32).tolist()))
        object.__setattr__(self, "init_mean", tuple(np.asarray(self.init_mean, dtype=F32).tolist()))
        object.__setattr__(self, "init_std", tuple(np.asarray(self.init_std, dtype=F32).tolist()))


@dataclass(frozen=True)
class SeparableNESState:
    """SeparableNES state — tensor leaves only, ``key`` last."""

    mean: SymbolicTensor
    sigma: SymbolicTensor
    weight: SymbolicTensor
    zero_mean_pop: SymbolicTensor
    best_fitness: SymbolicTensor
    key: SymbolicTensor


def _sn_init(config: SeparableNESConfig, key: SymbolicTensor) -> SeparableNESState:
    """Initial state: mean and per-dimension step sizes from init_std."""
    dim = len(config.init_mean)
    pop_size = config.pop_size
    mean = _constant_1d(config.init_mean)
    sigma = _constant_1d(config.init_std)
    if config.recombination_weights is None:
        weight = _default_recombination_weights(pop_size)
    else:
        weight = _constant_1d(config.recombination_weights)
    return SeparableNESState(
        mean=mean,
        sigma=sigma,
        weight=weight,
        zero_mean_pop=enp.zeros((pop_size, dim), dtype=F32),
        best_fitness=enp.full((), np.inf, dtype=F32),
        key=key,
    )


def _sn_ask(
    config: SeparableNESConfig, state: SeparableNESState
) -> tuple[SymbolicTensor, SeparableNESState]:
    """Sample the population with per-dimension (separable) step sizes."""
    pop_size = config.pop_size
    dim = len(config.init_mean)
    key, subkey = random.split(state.key)
    zero_mean_pop = random.normal(subkey, (pop_size, dim), mean=0.0, std=1.0, dtype=F32)
    population = state.mean + zero_mean_pop * state.sigma
    return population, SeparableNESState(
        mean=state.mean,
        sigma=state.sigma,
        weight=state.weight,
        zero_mean_pop=zero_mean_pop,
        best_fitness=state.best_fitness,
        key=key,
    )


def _sn_tell(
    config: SeparableNESConfig, state: SeparableNESState, fitness: SymbolicTensor
) -> SeparableNESState:
    """Natural-gradient update of the mean and per-dimension step sizes."""
    dim = len(config.init_mean)
    order = etl.argsort(fitness)
    zero_mean_pop = etl.gather(state.zero_mean_pop, order, axis=0)
    weight = etl.tile(enp.expand_dims(state.weight, axis=1), (1, dim))
    grad_mu = etl.sum(weight * zero_mean_pop, axes=0)
    grad_sigma = etl.sum(weight * (zero_mean_pop**2 - 1), axes=0)
    mean = state.mean + config.learning_rate_mean * state.sigma * grad_mu
    sigma = state.sigma * etl.exp(config.learning_rate_var / 2 * grad_sigma)
    best_fitness = etl.minimum(state.best_fitness, etl.min(fitness))
    return SeparableNESState(
        mean=mean,
        sigma=sigma,
        weight=state.weight,
        zero_mean_pop=state.zero_mean_pop,
        best_fitness=best_fitness,
        key=state.key,
    )


def init(config: Any, key: SymbolicTensor) -> Any:
    """Dispatch ``init`` on the (static) config type."""
    if isinstance(config, XNESConfig):
        return _xn_init(config, key)
    if isinstance(config, SeparableNESConfig):
        return _sn_init(config, key)
    raise TypeError(f"Unknown NES config type: {type(config)!r}")


def ask(config: Any, state: Any) -> tuple[SymbolicTensor, Any]:
    """Dispatch ``ask`` on the (static) config type."""
    if isinstance(config, XNESConfig):
        return _xn_ask(config, state)
    if isinstance(config, SeparableNESConfig):
        return _sn_ask(config, state)
    raise TypeError(f"Unknown NES config type: {type(config)!r}")


def tell(config: Any, state: Any, fitness: SymbolicTensor) -> Any:
    """Dispatch ``tell`` on the (static) config type."""
    if isinstance(config, XNESConfig):
        return _xn_tell(config, state, fitness)
    if isinstance(config, SeparableNESConfig):
        return _sn_tell(config, state, fitness)
    raise TypeError(f"Unknown NES config type: {type(config)!r}")
