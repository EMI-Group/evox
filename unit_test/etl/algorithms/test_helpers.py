"""Tests for unit_test/etl/algorithms/helpers.py — the toy problems and the
generic init/ask/tell driver.  ETL-only, no torch.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import types
from dataclasses import dataclass

import numpy as np

import etl
import etl.numpy as enp
import etl.random as random

import helpers
from helpers import (
    AckleyConfig,
    DTLZ1Config,
    RosenbrockConfig,
    SphereConfig,
    ToyProblemState,
    run_generations,
    toy_evaluate,
)

N = 6
DIM = 7
F32 = np.dtype("float32")


def _evaluate(config, x):
    """Host helper: build + run toy_evaluate once for a numpy input."""
    spec = etl.core.TensorSpec(shape=tuple(x.shape), dtype=F32)
    exe = etl.build(toy_evaluate, config, ToyProblemState(), spec, backend="numpy")
    fitness, _ = etl.run(exe, config, ToyProblemState(), x)
    return fitness


def test_sphere_evaluate():
    fitness = _evaluate(SphereConfig(DIM), np.zeros((N, DIM), dtype=np.float32))
    assert fitness.shape == (N,)
    assert fitness.dtype == F32
    np.testing.assert_array_equal(np.asarray(fitness.numpy()), 0.0)


def test_rosenbrock_evaluate():
    fitness = _evaluate(RosenbrockConfig(DIM), np.zeros((N, DIM), dtype=np.float32))
    assert fitness.shape == (N,)
    assert fitness.dtype == F32
    np.testing.assert_allclose(np.asarray(fitness.numpy()), DIM - 1, rtol=1e-5)


def test_ackley_evaluate():
    fitness = _evaluate(AckleyConfig(DIM), np.zeros((N, DIM), dtype=np.float32))
    assert fitness.shape == (N,)
    assert fitness.dtype == F32
    np.testing.assert_allclose(np.asarray(fitness.numpy()), 0.0, atol=1e-5)


def test_dtlz1_matches_reference():
    rng = np.random.RandomState(1)
    x = rng.uniform(0.0, 1.0, size=(N, DIM)).astype(np.float32)
    n_obj = 3
    fitness = _evaluate(DTLZ1Config(DIM, n_obj), x)
    assert fitness.shape == (N, n_obj)
    assert fitness.dtype == F32
    # numpy reference of the torch DTLZ1 formula (minimize semantics)
    m = n_obj
    g = 100.0 * (
        DIM
        - m
        + 1
        + np.sum(
            (x[:, m - 1 :] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[:, m - 1 :] - 0.5)),
            axis=1,
            keepdims=True,
        )
    )
    fc = np.flip(np.cumprod(np.concatenate([np.ones((N, 1)), x[:, : m - 1]], axis=1), axis=1), axis=1)
    rp = np.concatenate([np.ones((N, 1)), 1.0 - np.flip(x[:, : m - 1], axis=1)], axis=1)
    expected = 0.5 * (1.0 + g) * fc * rp
    np.testing.assert_allclose(np.asarray(fitness.numpy()), expected, rtol=1e-5, atol=1e-5)


def test_toy_evaluate_outputs_finite():
    rng = np.random.RandomState(0)
    x = rng.uniform(-1.0, 1.0, size=(N, DIM)).astype(np.float32)
    for config in (
        SphereConfig(DIM),
        RosenbrockConfig(DIM),
        AckleyConfig(DIM),
        DTLZ1Config(DIM, 3),
    ):
        fitness = _evaluate(config, x)
        assert np.all(np.isfinite(np.asarray(fitness.numpy()))), type(config).__name__


# --- minimal inline toy algorithm (trivial random search) --------------------

@dataclass(frozen=True)
class ToyAlgoConfig:
    pop_size: int
    dim: int


@dataclass(frozen=True)
class ToyAlgoState:
    population: object
    best_fitness: object
    key: object


def toy_init(config, key):
    key1, key2 = random.split(key)
    pop = random.uniform(
        key1, (config.pop_size, config.dim), low=-5.0, high=5.0, dtype=F32
    )
    best = enp.zeros((), dtype=F32) + etl.ops.constant(
        etl.core.tensor(np.asarray(np.inf, dtype=np.float32))
    )
    return ToyAlgoState(pop, best, key2)


def toy_ask(config, state):
    key1, key2 = random.split(state.key)
    noise = random.normal(key1, state.population.shape, mean=0.0, std=0.1, dtype=F32)
    return state.population + noise, ToyAlgoState(state.population, state.best_fitness, key2)


def toy_tell(config, state, fitness):
    min_fit = etl.min(fitness)
    best = etl.select(min_fit < state.best_fitness, min_fit, state.best_fitness)
    return ToyAlgoState(state.population, best, state.key)


TOY_ALGO = types.SimpleNamespace(init=toy_init, ask=toy_ask, tell=toy_tell)


def test_run_generations_smoke():
    algo_cfg = ToyAlgoConfig(pop_size=16, dim=4)
    final_state = run_generations(TOY_ALGO, algo_cfg, SphereConfig(4), n_gens=3, seed=0)
    best = float(np.asarray(final_state.best_fitness.numpy()))
    assert np.isfinite(best)          # the inf sentinel got replaced
    assert best <= np.inf             # never worse than the initial best
    assert best < 1000.0              # loose sanity: sphere on [-5, 5]**4 stays small


def test_run_generations_deterministic():
    algo_cfg = ToyAlgoConfig(pop_size=16, dim=4)
    s1 = run_generations(TOY_ALGO, algo_cfg, SphereConfig(4), n_gens=3, seed=7)
    s2 = run_generations(TOY_ALGO, algo_cfg, SphereConfig(4), n_gens=3, seed=7)
    np.testing.assert_array_equal(
        np.asarray(s1.best_fitness.numpy()), np.asarray(s2.best_fitness.numpy())
    )
    np.testing.assert_allclose(
        np.asarray(s1.population.numpy()), np.asarray(s2.population.numpy())
    )
