"""Property tests for evox_etl ``polynomial_mutation`` (ETL only, NO torch).

Checks determinism (same key + inputs → identical output), key divergence,
and the documented contract: output shape (n, d), float32, finite, and
within [lb, ub]. Also covers the out-of-bounds clamping, the pro_m=0
identity edge case and the baked-default (pro_m=1.0, dis_m=20.0) path.
Everything runs through ``etl.build`` + ``etl.run`` on the numpy backend.
"""

import numpy as np

import etl
from etl import core

from evox_etl.operators.mutation import polynomial_mutation

KEY_SPEC = core.TensorSpec((), np.dtype("int64"))
RNG = np.random.default_rng(0)

N, D = 64, 12

X = RNG.standard_normal((N, D)).astype(np.float32)
X_SPEC = core.TensorSpec((N, D), np.dtype("float32"))
LB = np.full(D, -1.0, dtype=np.float32)
UB = np.full(D, 1.0, dtype=np.float32)
BOUND_SPEC = core.TensorSpec((D,), np.dtype("float32"))


def make_key(seed: int) -> np.ndarray:
    """0-d int64 key array (numpy scalars are rejected at the run boundary)."""
    return np.array(seed, dtype=np.int64)


def to_numpy(out):
    """Convert etl.run results (etl tensor or tuple thereof) to numpy/python."""
    if isinstance(out, tuple):
        return tuple(o.numpy() if hasattr(o, "numpy") else o for o in out)
    return out.numpy() if hasattr(out, "numpy") else out


def build(fn, *specs):
    """Build an executable: tensor inputs as TensorSpec, statics passed as-is."""
    return etl.build(fn, *specs, backend="numpy")


def test_pm_shape_dtype_bounds_finite():
    exe = build(polynomial_mutation, KEY_SPEC, X_SPEC, BOUND_SPEC, BOUND_SPEC, 1.0, 20.0)
    out = to_numpy(etl.run(exe, make_key(0), X, LB, UB, 1.0, 20.0))
    assert out.shape == (N, D) and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert np.all(out >= LB - 1e-5) and np.all(out <= UB + 1e-5)


def test_pm_clamps_out_of_bounds_input():
    x = RNG.uniform(-3.0, 3.0, (N, D)).astype(np.float32)
    exe = build(polynomial_mutation, KEY_SPEC, X_SPEC, BOUND_SPEC, BOUND_SPEC, 1.0, 20.0)
    out = to_numpy(etl.run(exe, make_key(0), x, LB, UB, 1.0, 20.0))
    assert np.all(out >= LB - 1e-5) and np.all(out <= UB + 1e-5)


def test_pm_pro_m_zero_is_identity_inside_bounds():
    # pro_m=0 → no site is mutated → output is exactly the clamped input
    # (identical to the input when the input is already within [lb, ub])
    x_in = np.clip(X, LB, UB)
    exe = build(polynomial_mutation, KEY_SPEC, X_SPEC, BOUND_SPEC, BOUND_SPEC, 0.0, 20.0)
    out = to_numpy(etl.run(exe, make_key(0), x_in, LB, UB, 0.0, 20.0))
    np.testing.assert_array_equal(out, x_in)


def test_pm_determinism_and_divergence():
    exe = build(polynomial_mutation, KEY_SPEC, X_SPEC, BOUND_SPEC, BOUND_SPEC, 1.0, 20.0)
    a = to_numpy(etl.run(exe, make_key(1), X, LB, UB, 1.0, 20.0))
    b = to_numpy(etl.run(exe, make_key(1), X, LB, UB, 1.0, 20.0))
    c = to_numpy(etl.run(exe, make_key(2), X, LB, UB, 1.0, 20.0))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_pm_defaults_baked():
    # pro_m/dis_m omitted at build AND at run → defaults (1.0, 20.0) baked in
    exe = build(polynomial_mutation, KEY_SPEC, X_SPEC, BOUND_SPEC, BOUND_SPEC)
    out = to_numpy(etl.run(exe, make_key(3), X, LB, UB))
    ref_exe = build(polynomial_mutation, KEY_SPEC, X_SPEC, BOUND_SPEC, BOUND_SPEC, 1.0, 20.0)
    ref = to_numpy(etl.run(ref_exe, make_key(3), X, LB, UB, 1.0, 20.0))
    assert np.array_equal(out, ref)
