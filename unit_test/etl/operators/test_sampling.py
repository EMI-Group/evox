"""Property tests for the functional evox_etl sampling operators (ETL only, NO torch).

Covers the deterministic Das-Dennis/grid samplers (``uniform_sampling``,
``grid_sampling``) and the keyed Latin-hypercube samplers
(``latin_hypercube_sampling_standard``, ``latin_hypercube_sampling``). Every
keyed operator is checked for determinism (same key + inputs → identical
output), key divergence (a different key → different output) and its
documented shape/dtype/bounds contract. Everything runs through
``etl.build`` + ``etl.run`` on the numpy backend (ETL has no eager mode).

Note on ``latin_hypercube_sampling_standard(smooth=False)``: the cell
PERMUTATION is still key-dependent (the permutation draw always happens —
same as torch), so different keys give different ROW ORDERS. What is
deterministic across keys is the SET of cell centers per column: each column
is a permutation of ``{(c + 0.5) / n : c in [0, n)}``.
"""

import numpy as np
import pytest

import etl
from etl import core

from evox_etl.operators.sampling import (
    grid_sampling,
    latin_hypercube_sampling,
    latin_hypercube_sampling_standard,
    uniform_sampling,
)

KEY_SPEC = core.TensorSpec((), np.dtype("int64"))
RNG = np.random.default_rng(0)


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


# --- uniform_sampling (Das-Dennis, deterministic) ---------------------------


@pytest.mark.parametrize("n,m", [(20, 3), (30, 5), (6, 4), (100, 2)])
def test_uniform_sampling_shape_dtype_bounds(n, m):
    exe = build(uniform_sampling, n, m)
    w, n_samples = to_numpy(etl.run(exe, n, m))
    assert isinstance(n_samples, int) and n_samples == w.shape[0]
    assert n_samples <= n
    assert w.shape == (n_samples, m) and w.dtype == np.float32
    assert np.all(w >= 1e-6 - 1e-9) and np.all(w <= 1.0 + 1e-9)
    # rows are convex-combination weights (up to the 1e-6 floor clamp)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_uniform_sampling_includes_unit_vectors():
    # the Das-Dennis set contains e_j (one objective = 1, rest clamped to 1e-6)
    exe = build(uniform_sampling, 20, 3)
    w, _ = to_numpy(etl.run(exe, 20, 3))
    for j in range(3):
        e_j = np.full(3, 1e-6, dtype=np.float32)
        e_j[j] = 1.0
        assert np.any(np.all(np.isclose(w, e_j, atol=1e-6), axis=1))


def test_uniform_sampling_deterministic():
    exe = build(uniform_sampling, 30, 5)
    w1, n1 = to_numpy(etl.run(exe, 30, 5))
    w2, n2 = to_numpy(etl.run(exe, 30, 5))
    assert n1 == n2 and np.array_equal(w1, w2)


# --- grid_sampling (deterministic) ------------------------------------------


@pytest.mark.parametrize("n,m", [(20, 3), (30, 5), (6, 4)])
def test_grid_sampling_shape_dtype_grid_values(n, m):
    exe = build(grid_sampling, n, m)
    w, n_samples = to_numpy(etl.run(exe, n, m))
    num_points = int(np.ceil(n ** (1 / m)))
    assert isinstance(n_samples, int) and n_samples == w.shape[0] == num_points ** m
    assert w.shape == (n_samples, m) and w.dtype == np.float32
    assert np.all(w >= 0.0) and np.all(w <= 1.0)
    # each column is an axis of the grid: the multiset of values equals the
    # linspace gap, each value repeated num_points**(m-1) times
    gap = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    expected_col = np.repeat(gap, num_points ** (m - 1))
    for j in range(m):
        np.testing.assert_allclose(
            np.sort(w[:, j]), np.sort(expected_col), rtol=1e-6, atol=1e-6
        )


def test_grid_sampling_deterministic():
    exe = build(grid_sampling, 20, 3)
    w1, n1 = to_numpy(etl.run(exe, 20, 3))
    w2, n2 = to_numpy(etl.run(exe, 20, 3))
    assert n1 == n2 and np.array_equal(w1, w2)


# --- latin_hypercube_sampling_standard ---------------------------------------


@pytest.mark.parametrize("smooth", [True, False], ids=["smooth", "cell-centers"])
def test_lhs_standard_shape_dtype_bounds(smooth):
    n, d = 32, 8
    exe = build(latin_hypercube_sampling_standard, KEY_SPEC, n, d, smooth)
    out = to_numpy(etl.run(exe, make_key(0), n, d, smooth))
    assert out.shape == (n, d) and out.dtype == np.float32
    assert np.all(out >= 0.0) and np.all(out < 1.0)


def test_lhs_standard_latin_property_smooth():
    # latin property: each cell index appears exactly once per column
    n, d = 16, 5
    exe = build(latin_hypercube_sampling_standard, KEY_SPEC, n, d, True)
    out = to_numpy(etl.run(exe, make_key(1), n, d, True))
    cells = np.floor(out * n).astype(np.int64)
    for j in range(d):
        assert sorted(cells[:, j].tolist()) == list(range(n))


def test_lhs_standard_cell_centers_deterministic_across_keys():
    # smooth=False: the multiset of values per column is the fixed cell-center
    # set {(c + 0.5) / n} — identical for every key. (The row ORDER is not
    # deterministic across keys; the permutation draw still happens, as in
    # torch.)
    n, d = 16, 5
    exe = build(latin_hypercube_sampling_standard, KEY_SPEC, n, d, False)
    centers = (np.arange(n, dtype=np.float64) + 0.5) / n
    for seed in (0, 1, 2):
        out = to_numpy(etl.run(exe, make_key(seed), n, d, False))
        for j in range(d):
            np.testing.assert_allclose(np.sort(out[:, j]), centers, rtol=0, atol=1e-6)


@pytest.mark.parametrize("smooth", [True, False], ids=["smooth", "cell-centers"])
def test_lhs_standard_determinism(smooth):
    n, d = 16, 5
    exe = build(latin_hypercube_sampling_standard, KEY_SPEC, n, d, smooth)
    a = to_numpy(etl.run(exe, make_key(7), n, d, smooth))
    b = to_numpy(etl.run(exe, make_key(7), n, d, smooth))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("smooth", [True, False], ids=["smooth", "cell-centers"])
def test_lhs_standard_key_divergence(smooth):
    n, d = 16, 5
    exe = build(latin_hypercube_sampling_standard, KEY_SPEC, n, d, smooth)
    a = to_numpy(etl.run(exe, make_key(7), n, d, smooth))
    b = to_numpy(etl.run(exe, make_key(8), n, d, smooth))
    assert not np.array_equal(a, b)


def test_lhs_standard_default_smooth_baked():
    # smooth omitted at build AND at run → the Python default (True) is baked
    # into the executable and must NOT be re-passed (issue 17)
    n, d = 16, 5
    exe = build(latin_hypercube_sampling_standard, KEY_SPEC, n, d)
    out = to_numpy(etl.run(exe, make_key(9), n, d))
    ref_exe = build(latin_hypercube_sampling_standard, KEY_SPEC, n, d, True)
    ref = to_numpy(etl.run(ref_exe, make_key(9), n, d, True))
    assert np.array_equal(out, ref)


# --- latin_hypercube_sampling (bounded) --------------------------------------


@pytest.mark.parametrize("smooth", [True, False], ids=["smooth", "cell-centers"])
def test_lhs_bounded_shape_dtype_bounds(smooth):
    n, d = 32, 6
    lb = RNG.uniform(-5.0, 0.0, d).astype(np.float32)
    ub = RNG.uniform(0.5, 5.0, d).astype(np.float32)
    bound_spec = core.TensorSpec((d,), np.dtype("float32"))
    exe = build(latin_hypercube_sampling, KEY_SPEC, n, bound_spec, bound_spec, smooth)
    out = to_numpy(etl.run(exe, make_key(0), n, lb, ub, smooth))
    assert out.shape == (n, d) and out.dtype == np.float32
    # samples in [0, 1) scaled row-wise into [lb, ub]
    assert np.all(out >= lb - 1e-6) and np.all(out <= ub + 1e-6)


def test_lhs_bounded_determinism_and_divergence():
    n, d = 16, 5
    lb = np.full(d, -2.0, dtype=np.float32)
    ub = np.full(d, 2.0, dtype=np.float32)
    bound_spec = core.TensorSpec((d,), np.dtype("float32"))
    exe = build(latin_hypercube_sampling, KEY_SPEC, n, bound_spec, bound_spec, True)
    a = to_numpy(etl.run(exe, make_key(3), n, lb, ub, True))
    b = to_numpy(etl.run(exe, make_key(3), n, lb, ub, True))
    c = to_numpy(etl.run(exe, make_key(4), n, lb, ub, True))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
