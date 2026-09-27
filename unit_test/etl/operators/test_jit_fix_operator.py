"""Property tests for the jit-fix operator utils (ETL only, NO torch).

Covers the torch-``torch.where``/``torch.clamp``-fusion replacements added to
``evox_etl/operators/jit_fix_operator.py``: ``switch``, ``clip``,
``maximum_float`` and ``minimum_float``. Like every plain function in that
module these are trace-only (ETL has no eager mode), so each one is driven
inside an ``etl.build`` + ``etl.run`` graph on the numpy backend. Exact
torch-parity for the same four helpers lives in ``parity/`` (torch imports are
allowed only there); here we assert their documented contracts:
``switch`` selects per label from a list of tensors, ``clip`` maps values into
[0, 1], and the float max/min compositions behave elementwise.
"""

import etl
import numpy as np
from etl import core

from evox_etl.operators.jit_fix_operator import (
    clip,
    maximum_float,
    minimum_float,
    switch,
)

F32 = np.dtype("float32")
I32 = np.dtype("int32")

VEC_SPEC = core.TensorSpec((5,), F32)
LABEL_SPEC = core.TensorSpec((5,), I32)

# values chosen to straddle both bounds and to be exactly representable
CLIP_IN = np.array([-0.5, 0.0, 0.3, 1.0, 2.0], dtype=np.float32)


def build(fn, *specs):
    """Build an executable on the numpy backend (statics passed as-is)."""
    return etl.build(fn, *specs, backend="numpy")


def to_numpy(out):
    """Convert etl.run results (etl tensor or tuple thereof) to numpy."""
    if isinstance(out, tuple):
        return tuple(o.numpy() if hasattr(o, "numpy") else o for o in out)
    return out.numpy() if hasattr(out, "numpy") else out


# --- switch ------------------------------------------------------------------


def test_switch_selects_per_label():
    labels = np.array([0, 1, 2, 2, 1], dtype=np.int32)
    values = [
        np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float32),
        np.array([20.0, 21.0, 22.0, 23.0, 24.0], dtype=np.float32),
        np.array([30.0, 31.0, 32.0, 33.0, 34.0], dtype=np.float32),
    ]
    exe = build(switch, LABEL_SPEC, [VEC_SPEC] * 3)
    out = to_numpy(etl.run(exe, labels, values))
    assert out.shape == (5,) and out.dtype == np.float32
    expected = np.array([10.0, 21.0, 32.0, 33.0, 24.0], dtype=np.float32)
    np.testing.assert_array_equal(out, expected)


def test_switch_two_values_is_a_single_select():
    labels = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    a = np.zeros(5, dtype=np.float32)
    b = np.ones(5, dtype=np.float32)
    exe = build(switch, LABEL_SPEC, [VEC_SPEC] * 2)
    out = to_numpy(etl.run(exe, labels, [a, b]))
    np.testing.assert_array_equal(out, labels.astype(np.float32))


def test_switch_deterministic_repeat():
    labels = np.array([2, 0, 1, 2, 0], dtype=np.int32)
    values = [np.full(5, float(k), dtype=np.float32) for k in range(3)]
    exe = build(switch, LABEL_SPEC, [VEC_SPEC] * 3)
    a = to_numpy(etl.run(exe, labels, values))
    b = to_numpy(etl.run(exe, labels, values))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, np.array([2.0, 0.0, 1.0, 2.0, 0.0], dtype=np.float32))


# --- clip --------------------------------------------------------------------


def test_clip_maps_into_unit_interval():
    exe = build(clip, VEC_SPEC)
    out = to_numpy(etl.run(exe, CLIP_IN))
    assert out.shape == (5,) and out.dtype == np.float32
    np.testing.assert_allclose(out, np.array([0.0, 0.0, 0.3, 1.0, 1.0], dtype=np.float32))


def test_clip_leaves_interior_values_untouched():
    x = np.linspace(0.05, 0.95, 5, dtype=np.float32)
    exe = build(clip, VEC_SPEC)
    out = to_numpy(etl.run(exe, x))
    np.testing.assert_array_equal(out, x)


def test_clip_is_idempotent():
    exe = build(clip, VEC_SPEC)
    once = to_numpy(etl.run(exe, CLIP_IN))
    twice = to_numpy(etl.run(exe, once))
    np.testing.assert_array_equal(once, twice)


# --- maximum_float / minimum_float -------------------------------------------


def test_maximum_float_against_scalar():
    a = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    exe = build(maximum_float, VEC_SPEC, 0.5)
    out = to_numpy(etl.run(exe, a, 0.5))
    assert out.shape == (5,) and out.dtype == np.float32
    np.testing.assert_allclose(out, np.array([0.5, 0.5, 0.5, 1.0, 2.0], dtype=np.float32))


def test_minimum_float_against_scalar():
    a = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    exe = build(minimum_float, VEC_SPEC, 0.5)
    out = to_numpy(etl.run(exe, a, 0.5))
    assert out.shape == (5,) and out.dtype == np.float32
    np.testing.assert_allclose(out, np.array([-1.0, 0.0, 0.5, 0.5, 0.5], dtype=np.float32))


def test_maximum_minimum_float_identity_at_ties():
    # a == b elementwise → both compositions return `a` unchanged
    a = np.full(5, 0.25, dtype=np.float32)
    max_exe = build(maximum_float, VEC_SPEC, 0.25)
    min_exe = build(minimum_float, VEC_SPEC, 0.25)
    np.testing.assert_array_equal(to_numpy(etl.run(max_exe, a, 0.25)), a)
    np.testing.assert_array_equal(to_numpy(etl.run(min_exe, a, 0.25)), a)


def test_maximum_minimum_float_are_complementary():
    a = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    max_exe = build(maximum_float, VEC_SPEC, 0.5)
    min_exe = build(minimum_float, VEC_SPEC, 0.5)
    mx = to_numpy(etl.run(max_exe, a, 0.5))
    mn = to_numpy(etl.run(min_exe, a, 0.5))
    # min(a, b) + max(a, b) == a + b elementwise
    np.testing.assert_allclose(mn + mx, a + np.float32(0.5))
