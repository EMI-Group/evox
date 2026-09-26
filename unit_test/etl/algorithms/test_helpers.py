"""Tests for unit_test/etl/algorithms/helpers.py — the toy problems and the
generic step-protocol driver.  ETL-only, no torch.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import types

import etl
import numpy as np
from helpers import (
    AckleyConfig,
    DTLZ1Config,
    RosenbrockConfig,
    SphereConfig,
    ToyProblemState,
    run_generations,
    toy_evaluate,
)

from evox_etl.algorithms.so.de_variants import shade
from evox_etl.algorithms.so.pso_variants import pso

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


# --- step-protocol driver (run_generations) ----------------------------------
#
# Real converted modules stand in for the algorithm under test: ``pso``
# defines init_step/step (no final_step), ``shade`` only step.


def _pso_config(dim: int = 8, pop_size: int = 40):
    return pso.make_pso(pop_size=pop_size, lb=np.full(dim, -5.0), ub=np.full(dim, 5.0))


def test_run_generations_pso_converges():
    state = run_generations(pso, _pso_config(), SphereConfig(8), n_gens=30, seed=0)
    assert state.pop.shape == (40, 8)
    assert state.velocity.shape == (40, 8)
    assert state.fit.shape == (40,)
    assert state.global_best_location.shape == (8,)
    assert state.fit.dtype == F32
    assert np.isfinite(np.asarray(state.fit.numpy())).all()
    assert np.isfinite(np.asarray(state.pop.numpy())).all()
    best = float(state.global_best_fit.numpy())
    assert 0.0 <= best < 1.0  # PSO on [-5, 5]^8 Sphere: ~1e-2 after 30 gens


def test_run_generations_deterministic_rerun():
    s1 = run_generations(pso, _pso_config(), SphereConfig(8), n_gens=10, seed=3)
    s2 = run_generations(pso, _pso_config(), SphereConfig(8), n_gens=10, seed=3)
    np.testing.assert_array_equal(np.asarray(s1.pop.numpy()), np.asarray(s2.pop.numpy()))
    np.testing.assert_array_equal(np.asarray(s1.fit.numpy()), np.asarray(s2.fit.numpy()))
    np.testing.assert_array_equal(
        np.asarray(s1.velocity.numpy()), np.asarray(s2.velocity.numpy())
    )
    assert float(s1.global_best_fit.numpy()) == float(s2.global_best_fit.numpy())


def test_run_generations_zero_gens_returns_init_state():
    state = run_generations(pso, _pso_config(), SphereConfig(8), n_gens=0, seed=0)
    assert np.isinf(float(state.global_best_fit.numpy()))  # untouched init sentinel
    assert np.all(np.isfinite(np.asarray(state.pop.numpy())))


def test_run_generations_shade_converges_without_init_step():
    cfg = shade.make_shade(pop_size=16, lb=np.full(8, -5.0), ub=np.full(8, 5.0))
    early = run_generations(shade, cfg, SphereConfig(8), n_gens=5, seed=0)
    late = run_generations(shade, cfg, SphereConfig(8), n_gens=20, seed=0)
    early_best = float(np.asarray(early.fit.numpy()).min())
    late_best = float(np.asarray(late.fit.numpy()).min())
    assert np.isfinite(early_best) and np.isfinite(late_best)
    assert late_best < early_best
    assert late_best < 10.0


def _recorded(fn, calls):
    """Wrap ``fn``, appending one entry per INVOCATION.

    ``etl.build`` traces a function exactly once per build (later ``etl.run``
    calls replay the recorded graph without re-invoking Python), so with the
    driver's per-(function, shape) exe cache ``len(calls)`` equals the number
    of distinct builds of ``fn``.  That is exactly what dispatch verification
    needs: generation 0 traced through ``init_step`` and later generations
    through ``step`` appear as one build each.
    """

    def wrapper(*args, **kwargs):
        calls.append(None)
        return fn(*args, **kwargs)

    return wrapper


def test_init_step_used_for_generation_zero():
    init_calls, init_step_calls, step_calls = [], [], []
    mod = types.SimpleNamespace(
        init=_recorded(pso.init, init_calls),
        init_step=_recorded(pso.init_step, init_step_calls),
        step=_recorded(pso.step, step_calls),
    )
    run_generations(mod, _pso_config(), SphereConfig(8), n_gens=4, seed=0)
    assert len(init_calls) == 1
    assert len(init_step_calls) == 1  # generation 0 traced via init_step
    assert len(step_calls) == 1  # generations 1-3 reuse ONE step exe (shape cache)


def test_step_fallback_when_module_has_no_init_step():
    init_calls, step_calls = [], []
    mod = types.SimpleNamespace(
        init=_recorded(shade.init, init_calls),
        step=_recorded(shade.step, step_calls),
    )
    cfg = shade.make_shade(pop_size=16, lb=np.full(8, -5.0), ub=np.full(8, 5.0))
    state = run_generations(mod, cfg, SphereConfig(8), n_gens=4, seed=0)
    assert len(init_calls) == 1
    assert len(step_calls) == 1  # generation 0 fell back to step
    assert np.all(np.isfinite(np.asarray(state.fit.numpy())))


def test_final_step_used_for_last_generation():
    init_step_calls, step_calls, final_calls = [], [], []
    mod = types.SimpleNamespace(
        init=pso.init,
        init_step=_recorded(pso.init_step, init_step_calls),
        step=_recorded(pso.step, step_calls),
        # PSO defines no final_step; stand in with its step (same signature).
        final_step=_recorded(pso.step, final_calls),
    )

    def counts():
        return (len(init_step_calls), len(step_calls), len(final_calls))

    run_generations(mod, _pso_config(), SphereConfig(8), n_gens=4, seed=0)
    assert counts() == (1, 1, 1)  # init_step | two steps | final_step

    for calls in (init_step_calls, step_calls, final_calls):
        calls.clear()
    run_generations(mod, _pso_config(), SphereConfig(8), n_gens=2, seed=0)
    assert counts() == (1, 0, 1)  # no middle generation -> plain step never built

    for calls in (init_step_calls, step_calls, final_calls):
        calls.clear()
    run_generations(mod, _pso_config(), SphereConfig(8), n_gens=1, seed=0)
    assert counts() == (1, 0, 0)  # single generation: init_step only
