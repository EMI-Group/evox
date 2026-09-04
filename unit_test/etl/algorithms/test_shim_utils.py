"""Tests for the jit-fix utility helpers (``clamp``/``lexsort``/``nanmin``/...).

Converted from the deprecated ``evox_etl.algorithms._shim_utils`` compat stub;
now imports the canonical ``evox_etl.operators.jit_fix_operator`` module.
"""
import sys
from pathlib import Path
sys.path[0:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[3] / "src")]

import numpy as np
import etl
from etl import core

from evox_etl.operators.jit_fix_operator import (
    clamp, clamp_float, clamp_int, lexsort, maximum, maximum_int,
    minimum, minimum_int, nanmax, nanmin, randint,
)


def _exercise(key, a, lb, ub, keys, with_nan):
    k1, k2 = etl.random.split(key)
    return (
        clamp(a, lb, ub),
        clamp_float(a, 0.0, 2.0),
        clamp_int(a, -1, 1),
        maximum(a, lb), minimum(a, ub),
        maximum_int(a, 0), minimum_int(a, 0),
        lexsort(keys),
        nanmin(with_nan)[0], nanmin(with_nan)[1],
        nanmax(with_nan)[0], nanmax(with_nan)[1],
        randint(k1, 3, 8, (7,), dtype=etl.int32),
        randint(k2, 0, 10, (5,), dtype=etl.int32),
    )


A = np.array([[-3.0, 0.5, 4.0, 2.5],
              [1.0, -2.0, 0.0, 3.0],
              [5.0, 1.5, -1.0, 0.5]], dtype=np.float32)
LB = np.full(4, -1.0, dtype=np.float32)
UB = np.full(4, 2.0, dtype=np.float32)
K0 = np.array([2.0, 0.0, 1.0, 0.0], dtype=np.float32)
K1 = np.array([9.0, 7.0, 7.0, 8.0], dtype=np.float32)
WITH_NAN = np.array([[1.0, 2.0, np.nan, 0.5],
                     [np.nan, np.nan, np.nan, np.nan],
                     [0.5, np.nan, 3.0, 1.0]], dtype=np.float32)


def test_jit_fix_utils():
    exe = etl.build(
        _exercise,
        core.TensorSpec(shape=(), dtype=np.dtype("int64")),
        core.TensorSpec(shape=(3, 4), dtype=np.dtype("float32")),
        core.TensorSpec(shape=(4,), dtype=np.dtype("float32")),
        core.TensorSpec(shape=(4,), dtype=np.dtype("float32")),
        [core.TensorSpec(shape=(4,), dtype=np.dtype("float32")),
         core.TensorSpec(shape=(4,), dtype=np.dtype("float32"))],
        core.TensorSpec(shape=(3, 4), dtype=np.dtype("float32")),
        backend="numpy",
    )
    outs = etl.run(exe, np.array(42, dtype=np.int64), A, LB, UB, [K0, K1], WITH_NAN)
    c, cf, ci, mx, mn, mxi, mni, ls, nmin_v, nmin_i, nmax_v, nmax_i, r1, r2 = [o.numpy() for o in outs]

    np.testing.assert_allclose(c, np.clip(A, -1.0, 2.0), rtol=0, atol=1e-6)
    np.testing.assert_allclose(cf, np.clip(A, 0.0, 2.0), rtol=0, atol=1e-6)
    np.testing.assert_allclose(ci, np.clip(A, -1, 1), rtol=0, atol=1e-6)
    np.testing.assert_allclose(mx, np.maximum(A, LB), rtol=0, atol=1e-6)
    np.testing.assert_allclose(mn, np.minimum(A, UB), rtol=0, atol=1e-6)
    np.testing.assert_allclose(mxi, np.maximum(A, 0), rtol=0, atol=1e-6)
    np.testing.assert_allclose(mni, np.minimum(A, 0), rtol=0, atol=1e-6)

    # lexsort: primary key K1, tie (values 7.0 at idx 1 and 2) broken by K0 (0.0 < 1.0)
    np.testing.assert_array_equal(ls, np.lexsort((K0, K1)))

    # nanmin/nanmax: NaN ignored; all-NaN row -> +-inf with index 0
    np.testing.assert_allclose(nmin_v, [0.5, np.inf, 0.5], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(nmin_i, [3, 0, 0])
    np.testing.assert_allclose(nmax_v, [2.0, -np.inf, 3.0], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(nmax_i, [1, 0, 2])

    # randint: range + determinism
    assert r1.shape == (7,) and np.issubdtype(r1.dtype, np.integer)
    assert np.all((r1 >= 3) & (r1 < 8))
    assert r2.shape == (5,) and np.all((r2 >= 0) & (r2 < 10))
    outs2 = etl.run(exe, np.array(42, dtype=np.int64), A, LB, UB, [K0, K1], WITH_NAN)
    assert np.array_equal(outs2[12].numpy(), r1) and np.array_equal(outs2[13].numpy(), r2)
