"""Tests for the canonical non-dominated selection operators (no torch).

Converted from loading the deprecated
``src/evox_etl/algorithms/_shim_selection_nd.py`` compat stub by path (a pure
re-export of the canonical module); now imports the canonical
``evox_etl.operators.selection`` operators directly. ``dominate_relation`` is
not exported at package level (mirrors torch), so it is imported from the
``non_dominate`` submodule. Outputs were checked bit-for-bit against the torch
reference implementation.

Tests use ETL only — no torch import.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import etl
from etl import core

from evox_etl.operators.selection import (
    crowding_distance,
    nd_environmental_selection,
    non_dominate_rank,
)
from evox_etl.operators.selection.non_dominate import dominate_relation

# Known 2-objective set; hand-computed non-domination ranks are [0, 0, 0, 0, 1, 2].
F = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [2, 4], [3, 5]], dtype=np.float32)
X = np.arange(12, dtype=np.float32).reshape(6, 2)  # arbitrary decision vectors


def spec(shape, dtype="float32"):
    return core.TensorSpec(shape=tuple(shape), dtype=np.dtype(dtype))


def build_run(fn, args, inputs):
    """Build fn with mixed TensorSpec/static args (fn's parameter order) and run."""
    exe = etl.build(fn, *args, backend="numpy")
    full_args = [
        next(inputs) if isinstance(a, core.TensorSpec) else a for a in args
    ]
    return etl.run(exe, *full_args)


def test_dominate_relation():
    drm = build_run(dominate_relation, [spec((6, 2)), spec((6, 2))], iter([F, F]))
    expected = np.array(
        [
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(np.asarray(drm.numpy()), expected)


def test_non_dominate_rank():
    rank = build_run(non_dominate_rank, [spec((6, 2))], iter([F]))
    np.testing.assert_array_equal(
        np.asarray(rank.numpy()), np.array([0, 0, 0, 0, 1, 2], dtype=np.int32)
    )


def test_crowding_distance_masked():
    mask = np.array([True, True, True, True, False, False])
    cd = np.asarray(
        build_run(
            crowding_distance,
            [spec((6, 2)), spec((6,), "bool")],
            iter([F, mask]),
        ).numpy(),
        dtype=np.float64,
    )
    # valid interior rows: 2/3 + 2/3; boundary rows: +inf; masked rows: -inf
    assert np.isposinf(cd[0]) and np.isposinf(cd[3])
    np.testing.assert_allclose(cd[1], 4.0 / 3.0, rtol=1e-6)
    np.testing.assert_allclose(cd[2], 4.0 / 3.0, rtol=1e-6)
    assert np.isneginf(cd[4]) and np.isneginf(cd[5])
    # masked rows are strictly last in any crowding-distance ordering
    assert np.all(cd[4:, None] < cd[:4])


def test_crowding_distance_no_mask():
    cd = np.asarray(
        build_run(crowding_distance, [spec((6, 2)), None], iter([F])).numpy(),
        dtype=np.float64,
    )
    assert np.isposinf(cd[0]) and np.isposinf(cd[3]) and np.isposinf(cd[5])
    np.testing.assert_allclose(cd[1], 5.0 / 6.0, rtol=1e-6)
    np.testing.assert_allclose(cd[2], 5.0 / 6.0, rtol=1e-6)
    np.testing.assert_allclose(cd[4], 7.0 / 12.0, rtol=1e-6)


def test_nd_environmental_selection():
    out = build_run(
        nd_environmental_selection,
        [spec((6, 2)), spec((6, 2)), 3],
        iter([X, F]),
    )
    x_sel, f_sel, rank_sel, cd_sel = (np.asarray(t.numpy()) for t in out)
    assert x_sel.shape == (3, 2) and f_sel.shape == (3, 2)
    assert rank_sel.shape == (3,) and cd_sel.shape == (3,)
    assert np.all(np.isfinite(x_sel)) and np.all(np.isfinite(f_sel))
    # selected solutions come from the rank-0 front only, in torch's exact order
    np.testing.assert_array_equal(rank_sel, np.zeros(3, dtype=np.int32))
    np.testing.assert_array_equal(x_sel, X[[0, 3, 1]])
    np.testing.assert_array_equal(f_sel, F[[0, 3, 1]])
    assert np.isposinf(cd_sel[0]) and np.isposinf(cd_sel[1])
    np.testing.assert_allclose(cd_sel[2], 4.0 / 3.0, rtol=1e-6)
