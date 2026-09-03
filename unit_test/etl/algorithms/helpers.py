"""Toy benchmark problems and a generic init/ask/tell smoke-test driver for the
ETL-based functional algorithms under test (no torch).

ETL host-side contract used throughout (verified empirically — do not
re-investigate):

* ``etl.build(fn, *specs)`` accepts plain callables; non-tensor arguments
  (config dataclasses, the empty ``ToyProblemState``) are recorded as STATIC
  values and are legal build arguments.
* ``etl.run(exe, *args)`` requires ALL arguments of the traced signature at
  every run — including the static ones (they are validated by value at the
  run boundary).  So config/state dataclasses are re-passed on each run.
* Scalar (shape=()) graph inputs must be 0-d numpy arrays
  (``np.asarray(v, dtype=np.int64)``) — numpy scalars are rejected.
* Reductions take an ``axes`` keyword (``etl.sum(x, axes=1)``), not ``axis``.
* Tensor slicing (``pop[:, 1:]``) and integer powers (``x ** 2``) work
  inside traces; ``enp.power`` exists for the general case.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

import etl
import etl.numpy as enp

__all__ = [
    "AckleyConfig",
    "DTLZ1Config",
    "RosenbrockConfig",
    "SphereConfig",
    "ToyProblemState",
    "run_generations",
    "toy_evaluate",
]


@dataclass(frozen=True)
class SphereConfig:
    """f(x) = sum(x**2); minimum 0 at the origin."""

    dim: int


@dataclass(frozen=True)
class RosenbrockConfig:
    """Classic valley function; minimum 0 at x = [1, ..., 1]."""

    dim: int


@dataclass(frozen=True)
class AckleyConfig:
    """Multi-modal; minimum 0 at the origin."""

    dim: int


@dataclass(frozen=True)
class DTLZ1Config:
    """DTLZ1 multi-objective problem (minimize); requires dim >= n_obj."""

    dim: int
    n_obj: int


@dataclass(frozen=True)
class ToyProblemState:
    """Empty problem state — the numerical toy problems are stateless."""


def toy_evaluate(config: Any, state: ToyProblemState, pop: Any) -> tuple[Any, ToyProblemState]:
    """Evaluate a toy benchmark on a population ``pop`` of shape (n, dim).

    The config is static at trace time, so Python ``isinstance`` dispatch is
    legal here.  Returns ``(fitness, state)``: fitness is (n,) float32 for the
    single-objective problems and (n, n_obj) float32 for DTLZ1.
    """
    if isinstance(config, SphereConfig):
        fitness = etl.sum(pop * pop, axes=1)
    elif isinstance(config, RosenbrockConfig):
        fitness = etl.sum(
            100.0 * (pop[:, 1:] - pop[:, :-1] ** 2) ** 2
            + (pop[:, :-1] - 1.0) ** 2,
            axes=1,
        )
    elif isinstance(config, AckleyConfig):
        fitness = _ackley(pop)
    elif isinstance(config, DTLZ1Config):
        fitness = _dtlz1(config, pop)
    else:
        raise TypeError(f"Unknown toy problem config: {type(config)!r}")
    return fitness, state


def _ackley(pop: Any) -> Any:
    """Ackley function (a=20, b=0.2, c=2*pi) — mirrors the torch evox reference."""
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    return (
        -a * enp.exp(-b * enp.sqrt(etl.mean(pop * pop, axes=1)))
        - enp.exp(etl.mean(enp.cos(c * pop), axes=1))
        + a
        + np.e
    )


def _dtlz1(config: DTLZ1Config, pop: Any) -> Any:
    """DTLZ1 (minimize) — mirrors ``src/evox/problems/numerical/dtlz.py``."""
    m = config.n_obj
    n, d = pop.shape
    f32 = np.dtype("float32")
    g = 100.0 * (
        d
        - m
        + 1
        + etl.sum(
            (pop[:, m - 1 :] - 0.5) ** 2 - enp.cos(20.0 * np.pi * (pop[:, m - 1 :] - 0.5)),
            axes=1,
            keepdims=True,
        )
    )
    flip_cumprod = etl.flip(
        etl.cumprod(
            etl.concatenate([enp.ones((n, 1), dtype=f32), pop[:, : m - 1]], axis=1),
            axis=1,
        ),
        axes=[1],
    )
    rest_part = etl.concatenate(
        [enp.ones((n, 1), dtype=f32), 1.0 - etl.flip(pop[:, : m - 1], axes=[1])],
        axis=1,
    )
    return 0.5 * (1.0 + g) * flip_cumprod * rest_part


def _leaf_spec(t: Any) -> "etl.core.TensorSpec":
    """TensorSpec mirroring an etl tensor leaf."""
    return etl.core.TensorSpec(shape=tuple(t.shape), dtype=np.dtype(t.dtype))


def _spec_tree(pytree: Any) -> Any:
    """Pytree of etl tensors -> pytree of TensorSpecs."""
    return etl.tree_map(_leaf_spec, pytree)


def _spec_key(*values: Any) -> tuple:
    """Hashable identity of the tensor-leaf shapes/dtypes of the given pytrees."""
    leaves: list[Any] = []
    for value in values:
        leaves.extend(etl.tree_leaves(value))
    return tuple((tuple(leaf.shape), str(leaf.dtype)) for leaf in leaves)


def run_generations(
    algo_mod: Any,
    algo_cfg: Any,
    prob_cfg: Any,
    n_gens: int,
    seed: int = 0,
) -> Any:
    """Drive a functional init/ask/tell algorithm module for ``n_gens``
    generations on a toy problem; return the final algorithm state.

    ``algo_mod`` must expose plain functions ``init(config, key) -> state``,
    ``ask(config, state) -> (candidates, state)`` and
    ``tell(config, state, fitness) -> state``.  When BOTH ``init_ask`` and
    ``init_tell`` are present they are used for generation 0 instead (NSGA
    style).

    All executables use the ``"numpy"`` backend; each distinct
    (function, tensor-shape) combination is built once and cached, so shapes
    changing between generations (e.g. NSGA-style first generation) only cost
    an extra build.
    """
    init_exe = etl.build(
        algo_mod.init,
        algo_cfg,
        etl.core.TensorSpec(shape=(), dtype=np.dtype("int64")),
        backend="numpy",
    )
    state = etl.run(init_exe, algo_cfg, np.asarray(seed, dtype=np.int64))

    exe_cache: dict[tuple, Any] = {}

    def build_cached(key: tuple, fn: Any, *specs: Any) -> Any:
        exe = exe_cache.get(key)
        if exe is None:
            exe = etl.build(fn, *specs, backend="numpy")
            exe_cache[key] = exe
        return exe

    for gen in range(n_gens):
        if (
            gen == 0
            and callable(getattr(algo_mod, "init_ask", None))
            and callable(getattr(algo_mod, "init_tell", None))
        ):
            ask_fn, tell_fn = algo_mod.init_ask, algo_mod.init_tell
        else:
            ask_fn, tell_fn = algo_mod.ask, algo_mod.tell

        ask_exe = build_cached(
            (id(ask_fn), _spec_key(state)),
            ask_fn,
            algo_cfg,
            _spec_tree(state),
        )
        candidates, state = etl.run(ask_exe, algo_cfg, state)

        eval_exe = build_cached(
            (id(toy_evaluate), _spec_key(candidates)),
            toy_evaluate,
            prob_cfg,
            ToyProblemState(),
            _leaf_spec(candidates),
        )
        fitness, _ = etl.run(eval_exe, prob_cfg, ToyProblemState(), candidates)

        tell_exe = build_cached(
            (id(tell_fn), _spec_key(state, fitness)),
            tell_fn,
            algo_cfg,
            _spec_tree(state),
            _leaf_spec(fitness),
        )
        state = etl.run(tell_exe, algo_cfg, state, fitness)

    return state
