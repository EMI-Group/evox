"""Pure-etl tests for the DTLZ problems (NO torch imports here).

Covers evaluate shapes/dtypes for all variants at their default configs, known
values, custom configs, and the ``pf`` reference fronts (shapes + value
ranges).  Everything runs through ``etl.build`` + ``etl.run`` on the default
numpy backend; traced shapes are fully static (concrete n and d).
"""

import numpy as np
import pytest

import etl
import etl.core

from evox_etl.problems.numerical.dtlz import (
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ6,
    DTLZ7,
    evaluate,
    pf,
)
from evox_etl.problems.numerical.state import ProblemState

N = 64
M = 3
RNG = np.random.default_rng(42)

# (config class, default d) — mirrors torch evox defaults
DTLZ_CASES = [
    (DTLZ1, 7),
    (DTLZ2, 12),
    (DTLZ3, 12),
    (DTLZ4, 12),
    (DTLZ5, 12),
    (DTLZ6, 12),
    (DTLZ7, 21),
]


def run_evaluate(config, x):
    """Trace ``evaluate(config, ProblemState(), pop)`` and run it on ``x``."""
    spec = etl.core.TensorSpec(x.shape, np.dtype("float32"))
    exe = etl.build(evaluate, config, ProblemState(), spec)
    fitness, problem_state = etl.run(exe, config, ProblemState(), x)
    return fitness.numpy(), problem_state


def run_pf(config):
    """Trace ``pf(config)`` (no tensor args) and run it."""
    exe = etl.build(pf, config)
    return etl.run(exe, config).numpy()


# --- evaluate: shapes / dtypes / finiteness -------------------------------


@pytest.mark.parametrize("config_cls, d", DTLZ_CASES)
def test_evaluate_shapes_and_dtype(config_cls, d):
    config = config_cls()
    x = RNG.random((N, d)).astype(np.float32)
    fitness, problem_state = run_evaluate(config, x)
    assert fitness.shape == (N, M)
    assert fitness.dtype == np.float32
    assert np.all(np.isfinite(fitness))
    assert problem_state == ProblemState()


def test_evaluate_does_not_mutate_input():
    config = DTLZ2()
    x = RNG.random((N, config.d)).astype(np.float32)
    x_copy = x.copy()
    run_evaluate(config, x)
    np.testing.assert_array_equal(x, x_copy)


# --- known values (verified against the torch formulas) --------------------


def test_dtlz2_known_values_zero_front():
    # x[:, :m-1] = 0 and x[:, m-1:] = 0.5 gives g = 0, so
    # f = [cos(0)cos(0), cos(0)sin(0), sin(0)] = [1, 0, 0].
    config = DTLZ2()
    x = np.full((2, config.d), 0.5, dtype=np.float32)
    x[:, : config.m - 1] = 0.0
    fitness, _ = run_evaluate(config, x)
    np.testing.assert_allclose(fitness, [[1.0, 0.0, 0.0]] * 2, atol=1e-5)


def test_dtlz2_known_values_mid_front():
    # x = 0.5 everywhere gives g = 0 and theta = pi/4, so
    # f = [cos^2, cos*sin, sin] = [0.5, 0.5, sqrt(2)/2].
    config = DTLZ2()
    x = np.full((2, config.d), 0.5, dtype=np.float32)
    fitness, _ = run_evaluate(config, x)
    expected = np.array([0.5, 0.5, np.sqrt(2.0) / 2.0], dtype=np.float32)
    np.testing.assert_allclose(fitness, np.tile(expected, (2, 1)), atol=1e-5)


def test_dtlz1_known_values():
    # x[:, :m-1] = 0 and x[:, m-1:] = 0.5 makes every g term -1 and cancels to
    # g = 0, so f = 0.5 * flip(cumprod([1, 0, 0])) * [1, 1, 1] = [0, 0, 0.5].
    config = DTLZ1()
    x = np.full((2, config.d), 0.5, dtype=np.float32)
    x[:, : config.m - 1] = 0.0
    fitness, _ = run_evaluate(config, x)
    np.testing.assert_allclose(fitness, [[0.0, 0.0, 0.5]] * 2, atol=1e-5)


# --- config fields are respected -------------------------------------------


def test_custom_config_fields_respected():
    config = DTLZ2(d=13, m=4)
    x = RNG.random((8, 13)).astype(np.float32)
    fitness, _ = run_evaluate(config, x)
    assert fitness.shape == (8, 4)
    assert np.all(np.isfinite(fitness))


# --- pf reference fronts ----------------------------------------------------
# Row counts match the torch reference implementation exactly (same sampling
# math): Das-Dennis uniform sampling with n = ref_num * m = 3000, m = 3 yields
# C(77, 2) = 2926 rows (DTLZ1-4); the DTLZ5/6 pf has ref_num * m = 3000 rows;
# DTLZ7 grid sampling yields ceil(sqrt(3000))^2 = 3025 rows.


@pytest.mark.parametrize(
    "config_cls, n_rows, lo, hi",
    [
        (DTLZ1, 2926, 0.0, 0.5),
        (DTLZ2, 2926, 0.0, 1.0),
        (DTLZ3, 2926, 0.0, 1.0),  # inherits DTLZ2's pf
        (DTLZ4, 2926, 0.0, 1.0),  # inherits DTLZ2's pf
        (DTLZ5, 3000, 0.0, 1.0),
        (DTLZ6, 3000, 0.0, 1.0),
        (DTLZ7, 3025, 0.0, 6.0),
    ],
)
def test_pf_shapes_and_ranges(config_cls, n_rows, lo, hi):
    front = run_pf(config_cls())
    assert front.shape == (n_rows, M)
    assert front.dtype == np.float32
    assert np.all(np.isfinite(front))
    assert np.all(front >= lo - 1e-6)
    assert np.all(front <= hi + 1e-6)


def test_pf_config_fields_respected():
    front = run_pf(DTLZ2(m=4))
    assert front.shape[1] == 4
    assert np.all(np.isfinite(front))
