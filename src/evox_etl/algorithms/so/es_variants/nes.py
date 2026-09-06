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

from evox_etl.algorithms._config_utils import (
    ArrayLike,
    bake_float32_constant,
    require_gt,
    to_float_tuple,
)

F32 = np.dtype("float32")


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

    Dumb config: array fields are f32-rounded tuples of Python floats
    (``init_covar`` nested, one row-tuple per covariance row).  Use
    ``make_xnes`` for ndarray input, validation and eager derivation of the
    ``pop_size`` / learning-rate defaults from ``init_mean``'s dimension.
    """

    init_mean: tuple[float, ...]
    init_covar: tuple[tuple[float, ...], ...]
    pop_size: int | None = None
    recombination_weights: tuple[float, ...] | None = None
    learning_rate_mean: float | None = None
    learning_rate_var: float | None = None
    learning_rate_B: float | None = None
    covar_as_cholesky: bool = False


def _to_float_matrix(values: Any) -> tuple[tuple[float, ...], ...]:
    """Normalize a 2-D array-like to a tuple of f32-rounded float rows.

    ``init_covar`` is stored nested (one row-tuple per covariance row); the
    flat ``to_float_tuple`` cannot express that structure.
    """
    return tuple(tuple(row) for row in np.asarray(values, dtype=np.float32).tolist())


def make_xnes(
    init_mean: ArrayLike,
    init_covar: ArrayLike,
    pop_size: int | None = None,
    recombination_weights: ArrayLike | None = None,
    learning_rate_mean: float | None = None,
    learning_rate_var: float | None = None,
    learning_rate_B: float | None = None,
    covar_as_cholesky: bool = False,
) -> XNESConfig:
    """Construct an :class:`XNESConfig`, deriving defaults and validating input.

    Array arguments are normalized ONCE to f32-rounded tuples of Python
    floats; ``pop_size`` and the learning rates default from
    ``dim = len(init_mean)`` (mirroring the torch/etl derivation).  Raises
    ValueError on invalid input.
    """
    # NOTE: torch uses self.dim here while it is still undefined (bug) -> local dim.
    dim = np.asarray(init_mean).shape[0]
    if pop_size is None:
        pop_size = 4 + math.floor(3 * math.log(dim))
    require_gt("pop_size", pop_size, 0)
    if learning_rate_mean is None:
        learning_rate_mean = 1
    if learning_rate_var is None:
        learning_rate_var = (9 + 3 * math.log(dim)) / 5 / math.pow(dim, 1.5)
    if learning_rate_B is None:
        learning_rate_B = learning_rate_var
    require_gt("learning_rate_mean", learning_rate_mean, 0)
    require_gt("learning_rate_var", learning_rate_var, 0)
    require_gt("learning_rate_B", learning_rate_B, 0)
    if recombination_weights is not None:
        recombination_weights = to_float_tuple(recombination_weights, dtype=np.float32)
        w = np.asarray(recombination_weights)
        if not (w[1:] <= w[:-1]).all():
            raise ValueError(
                "recombination_weights must be in descending order, got "
                f"{recombination_weights!r}"
            )
    return XNESConfig(
        init_mean=to_float_tuple(init_mean, dtype=np.float32),
        init_covar=_to_float_matrix(init_covar),
        pop_size=pop_size,
        recombination_weights=recombination_weights,
        learning_rate_mean=learning_rate_mean,
        learning_rate_var=learning_rate_var,
        learning_rate_B=learning_rate_B,
        covar_as_cholesky=covar_as_cholesky,
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
    mean = bake_float32_constant(config.init_mean)
    covar = etl.ops.constant(
        etl.core.tensor(np.asarray(config.init_covar, dtype=F32))
    )
    if not config.covar_as_cholesky:
        covar = etl.cholesky(covar)
    if config.recombination_weights is None:
        weights = _default_recombination_weights(pop_size)
    else:
        weights = bake_float32_constant(config.recombination_weights)
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
    """SeparableNES hyperparameters (mirrors the torch ``__init__`` minus ``device``).

    Dumb config: array fields are f32-rounded tuples of Python floats; use
    ``make_separable_nes`` for ndarray input, validation and eager derivation
    of the ``pop_size`` / learning-rate defaults from ``init_mean``'s
    dimension.
    """

    init_mean: tuple[float, ...]
    init_std: tuple[float, ...]
    pop_size: int | None = None
    recombination_weights: tuple[float, ...] | None = None
    learning_rate_mean: float | None = None
    learning_rate_var: float | None = None


def make_separable_nes(
    init_mean: ArrayLike,
    init_std: ArrayLike,
    pop_size: int | None = None,
    recombination_weights: ArrayLike | None = None,
    learning_rate_mean: float | None = None,
    learning_rate_var: float | None = None,
) -> SeparableNESConfig:
    """Construct a :class:`SeparableNESConfig`, deriving defaults and validating.

    Array arguments are normalized ONCE to f32-rounded tuples of Python
    floats; ``pop_size`` and the learning rates default from
    ``dim = len(init_mean)`` (mirroring the torch/etl derivation).  Raises
    ValueError on invalid input.
    """
    # NOTE: torch uses self.dim here while it is still undefined (bug) -> local dim.
    dim = np.asarray(init_mean).shape[0]
    init_std_arr = np.asarray(init_std)
    if init_std_arr.shape != (dim,):
        raise ValueError(f"init_std must have shape ({dim},), got {init_std_arr.shape}")
    if pop_size is None:
        pop_size = 4 + math.floor(3 * math.log(dim))
    require_gt("pop_size", pop_size, 0)
    if learning_rate_mean is None:
        learning_rate_mean = 1
    if learning_rate_var is None:
        learning_rate_var = (3 + math.log(dim)) / 5 / math.sqrt(dim)
    require_gt("learning_rate_mean", learning_rate_mean, 0)
    require_gt("learning_rate_var", learning_rate_var, 0)
    if recombination_weights is not None:
        weights_arr = np.asarray(recombination_weights)
        if weights_arr.shape != (pop_size,):
            raise ValueError(
                f"recombination_weights must have shape ({pop_size},), got "
                f"{weights_arr.shape}"
            )
        recombination_weights = to_float_tuple(recombination_weights, dtype=np.float32)
    return SeparableNESConfig(
        init_mean=to_float_tuple(init_mean, dtype=np.float32),
        init_std=to_float_tuple(init_std, dtype=np.float32),
        pop_size=pop_size,
        recombination_weights=recombination_weights,
        learning_rate_mean=learning_rate_mean,
        learning_rate_var=learning_rate_var,
    )


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
    mean = bake_float32_constant(config.init_mean)
    sigma = bake_float32_constant(config.init_std)
    if config.recombination_weights is None:
        weight = _default_recombination_weights(pop_size)
    else:
        weight = bake_float32_constant(config.recombination_weights)
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
