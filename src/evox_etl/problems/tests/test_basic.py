"""Pure-etl tests for the basic numerical problems (NO torch imports here).

Covers shapes/dtypes, known optima, raw-math (no boundary clamping) behaviour,
shift/affine config handling and the plain ``*_func`` helpers.  Everything runs
through ``etl.build`` + ``etl.run`` on the default numpy backend.
"""

import numpy as np
import pytest

import etl
import etl.core

from evox_etl.problems.numerical.basic import (
    Ackley,
    Ellipsoid,
    Griewank,
    Levy,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    ShiftAffineNumericalProblem,
    Sphere,
    Zakharov,
    ackley_func,
    evaluate,
    levy_func,
    rosenbrock_func,
    sphere_func,
)
from evox_etl.problems.numerical.state import ProblemState

N = 64
DIM = 10
RNG = np.random.default_rng(0)


def run_evaluate(config, x):
    """Trace ``evaluate(config, ProblemState(), pop)`` and run it on ``x``."""
    spec = etl.core.TensorSpec(x.shape, np.dtype("float32"))
    exe = etl.build(evaluate, config, ProblemState(), spec)
    fitness, problem_state = etl.run(exe, config, ProblemState(), x)
    return fitness.numpy(), problem_state


def run_func(fn, x, *static_args):
    """Trace a plain ``*_func`` (e.g. ``ackley_func(a, b, c, x)``) and run it."""
    spec = etl.core.TensorSpec(x.shape, np.dtype("float32"))
    exe = etl.build(fn, *static_args, spec)
    return etl.run(exe, *static_args, x).numpy()


def zeros(n=N, dim=DIM):
    return np.zeros((n, dim), dtype=np.float32)


def ones(n=N, dim=DIM):
    return np.ones((n, dim), dtype=np.float32)


# --- shapes / dtypes -----------------------------------------------------


def test_sphere_shapes_and_dtype():
    fitness, problem_state = run_evaluate(Sphere(), zeros())
    assert fitness.shape == (N,)
    assert fitness.dtype == np.float32
    # the state passes through unchanged
    assert problem_state == ProblemState()


def test_population_not_mutated():
    x = RNG.standard_normal((N, DIM)).astype(np.float32)
    x_copy = x.copy()
    run_evaluate(Sphere(), x)
    np.testing.assert_array_equal(x, x_copy)


# --- known optima --------------------------------------------------------


@pytest.mark.parametrize(
    "config, x, expected",
    [
        (Sphere(), zeros(), 0.0),
        (Sphere(), ones(), DIM),
        (Rastrigin(), zeros(), 0.0),
        (Rosenbrock(), ones(), 0.0),
        (Ackley(), zeros(), 0.0),
        (Levy(), ones(), 0.0),
        (Griewank(), zeros(), 0.0),
        (Zakharov(), zeros(), 0.0),
        (Ellipsoid(), ones(), DIM * (DIM + 1) / 2),
    ],
)
def test_known_optima(config, x, expected):
    fitness, _ = run_evaluate(config, x)
    np.testing.assert_allclose(
        fitness, np.full(N, expected, dtype=np.float32), atol=1e-4, rtol=0.0
    )


def test_schwefel_optimum():
    x = np.full((N, DIM), 420.9687462275036, dtype=np.float32)
    fitness, _ = run_evaluate(Schwefel(), x)
    np.testing.assert_allclose(fitness, 0.0, atol=1e-2)


# --- raw math: no boundary handling --------------------------------------


def test_no_boundary_handling_raw_math():
    # torch evox's basic problems do NOT clamp: Sphere(1e2) = dim * 1e4.
    x = np.full((N, DIM), 1e2, dtype=np.float32)
    fitness, _ = run_evaluate(Sphere(), x)
    np.testing.assert_allclose(
        fitness, np.full(N, DIM * 1e4, dtype=np.float32), rtol=1e-5
    )


# --- shift / affine configs -----------------------------------------------


def test_shift_of_zeros_equals_unshifted():
    x = RNG.standard_normal((N, DIM)).astype(np.float32)
    f_shifted, _ = run_evaluate(Sphere(shift=np.zeros(DIM, dtype=np.float32)), x)
    f_plain, _ = run_evaluate(Sphere(), x)
    # shifting by an explicit zero vector must not change the result at all
    np.testing.assert_allclose(f_shifted, f_plain, rtol=0.0, atol=0.0)


def test_ackley_shift_equals_shifted_func():
    shift = np.ones(DIM, dtype=np.float32)
    x = RNG.standard_normal((N, DIM)).astype(np.float32)
    f_config, _ = run_evaluate(Ackley(shift=shift), x)
    # reference: build ackley_func itself on the shifted input
    f_reference = run_func(ackley_func, x + shift, 20.0, 0.2, 2 * np.pi)
    np.testing.assert_allclose(f_config, f_reference, rtol=1e-6, atol=1e-6)


def test_affine_identity_equals_unshifted():
    x = RNG.standard_normal((N, DIM)).astype(np.float32)
    f_affine, _ = run_evaluate(Sphere(affine=np.eye(DIM, dtype=np.float32)), x)
    f_plain, _ = run_evaluate(Sphere(), x)
    np.testing.assert_allclose(f_affine, f_plain, rtol=1e-6, atol=1e-6)


def test_shift_and_affine_combined():
    shift = np.ones(DIM, dtype=np.float32)
    affine = 2.0 * np.eye(DIM, dtype=np.float32)
    x = RNG.standard_normal((N, DIM)).astype(np.float32)
    f_config, _ = run_evaluate(Ackley(shift=shift, affine=affine), x)
    f_reference = run_func(ackley_func, (x + shift) @ affine, 20.0, 0.2, 2 * np.pi)
    np.testing.assert_allclose(f_config, f_reference, rtol=1e-6, atol=1e-6)


# --- the plain *_func helpers ---------------------------------------------


def test_sphere_func_direct():
    x = ones(n=8)
    np.testing.assert_allclose(
        run_func(sphere_func, x), np.full(8, DIM, dtype=np.float32), rtol=1e-6
    )


def test_ackley_func_signature_and_value():
    # signature is (a, b, c, x) — the scalars are static args passed positionally
    x = zeros(n=8)
    out = run_func(ackley_func, x, 20.0, 0.2, 2 * np.pi)
    np.testing.assert_allclose(out, 0.0, atol=1e-4)


def test_rosenbrock_and_levy_funcs_direct():
    x = ones(n=8)
    np.testing.assert_allclose(run_func(rosenbrock_func, x), 0.0, atol=1e-4)
    np.testing.assert_allclose(run_func(levy_func, x), 0.0, atol=1e-4)


# --- config validation (mirrors torch ShiftAffineNumericalProblem.__init__) ---


def test_invalid_affine_rejected():
    with pytest.raises(AssertionError, match="affine must be a square matrix"):
        Ackley(affine=np.ones((3, 2), dtype=np.float32))


def test_invalid_shift_rejected():
    with pytest.raises(AssertionError, match="shift must be a vector"):
        Ackley(shift=np.ones((2, 2), dtype=np.float32))


def test_mismatched_shift_affine_rejected():
    with pytest.raises(AssertionError, match="same dimension"):
        Ackley(shift=np.ones(3, dtype=np.float32), affine=np.eye(2, dtype=np.float32))


def test_bare_base_config_rejected():
    # the abstract base config dispatches to no true function
    with pytest.raises(TypeError, match="unknown numerical problem config"):
        run_evaluate(ShiftAffineNumericalProblem(), zeros())
