"""Property tests for the functional evox_etl selection operators (ETL only, NO torch).

Only the KEYED (random) selection operators are covered here —
``tournament_selection``, ``tournament_selection_multifit`` and
``select_rand_pbest``. The deterministic ones (``dominate_relation``,
``non_dominate_rank``, ``crowding_distance``, ``nd_environmental_selection``,
``apd_fn``, ``ref_vec_guided``) are covered by the torch parity suite in
``parity/``. Each operator is checked for determinism (same key + inputs →
identical output), key divergence, and its documented shape/dtype/bounds
contract, via ``etl.build`` + ``etl.run`` on the numpy backend.
"""

import numpy as np
import pytest

import etl
from etl import core

from evox_etl.operators.selection import (
    select_rand_pbest,
    tournament_selection,
    tournament_selection_multifit,
)

KEY_SPEC = core.TensorSpec((), np.dtype("int64"))
RNG = np.random.default_rng(0)

N = 64  # population size
N_ROUND = 100

# distinct, deterministic fitness values (no ties → argsort is unique)
FITNESS = np.arange(N, dtype=np.float32)
FIT_SPEC = core.TensorSpec((N,), np.dtype("float32"))
FITNESS2 = np.linspace(0.0, 1.0, N, dtype=np.float32)
POP = RNG.standard_normal((N, 10)).astype(np.float32)
POP_SPEC = core.TensorSpec((N, 10), np.dtype("float32"))


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


# --- tournament_selection ----------------------------------------------------


def test_tournament_shape_dtype_bounds():
    exe = build(tournament_selection, KEY_SPEC, N_ROUND, FIT_SPEC, 2)
    out = to_numpy(etl.run(exe, make_key(0), N_ROUND, FITNESS, 2))
    assert out.shape == (N_ROUND,) and out.dtype == np.int32
    assert np.all(out >= 0) and np.all(out < N)


def test_tournament_default_tournament_size_baked():
    # tournament_size omitted at build AND at run → the Python default (2) is
    # baked into the executable (issue 17) and matches the explicit variant
    exe = build(tournament_selection, KEY_SPEC, N_ROUND, FIT_SPEC)
    out = to_numpy(etl.run(exe, make_key(0), N_ROUND, FITNESS))
    ref_exe = build(tournament_selection, KEY_SPEC, N_ROUND, FIT_SPEC, 2)
    ref = to_numpy(etl.run(ref_exe, make_key(0), N_ROUND, FITNESS, 2))
    assert np.array_equal(out, ref)


def test_tournament_size_three():
    exe = build(tournament_selection, KEY_SPEC, N_ROUND, FIT_SPEC, 3)
    out = to_numpy(etl.run(exe, make_key(0), N_ROUND, FITNESS, 3))
    assert out.shape == (N_ROUND,) and out.dtype == np.int32
    assert np.all(out >= 0) and np.all(out < N)


def test_tournament_determinism():
    exe = build(tournament_selection, KEY_SPEC, N_ROUND, FIT_SPEC, 2)
    a = to_numpy(etl.run(exe, make_key(1), N_ROUND, FITNESS, 2))
    b = to_numpy(etl.run(exe, make_key(1), N_ROUND, FITNESS, 2))
    assert np.array_equal(a, b)


def test_tournament_key_divergence():
    exe = build(tournament_selection, KEY_SPEC, N_ROUND, FIT_SPEC, 2)
    a = to_numpy(etl.run(exe, make_key(1), N_ROUND, FITNESS, 2))
    b = to_numpy(etl.run(exe, make_key(2), N_ROUND, FITNESS, 2))
    assert not np.array_equal(a, b)


# --- tournament_selection_multifit -------------------------------------------


def _multifit_specs(n_fit):
    return [FIT_SPEC] * n_fit


def test_multifit_shape_dtype_bounds():
    fits = [FITNESS, FITNESS2]
    exe = build(tournament_selection_multifit, KEY_SPEC, N_ROUND, _multifit_specs(2), 2)
    out = to_numpy(etl.run(exe, make_key(0), N_ROUND, fits, 2))
    assert out.shape == (N_ROUND,) and out.dtype == np.int32
    assert np.all(out >= 0) and np.all(out < N)


def test_multifit_three_objectives():
    fits = [FITNESS, FITNESS2, np.linspace(1.0, 0.0, N, dtype=np.float32)]
    exe = build(tournament_selection_multifit, KEY_SPEC, N_ROUND, _multifit_specs(3), 2)
    out = to_numpy(etl.run(exe, make_key(0), N_ROUND, fits, 2))
    assert out.shape == (N_ROUND,) and out.dtype == np.int32
    assert np.all(out >= 0) and np.all(out < N)


def test_multifit_determinism_and_divergence():
    fits = [FITNESS, FITNESS2]
    exe = build(tournament_selection_multifit, KEY_SPEC, N_ROUND, _multifit_specs(2), 2)
    a = to_numpy(etl.run(exe, make_key(3), N_ROUND, fits, 2))
    b = to_numpy(etl.run(exe, make_key(3), N_ROUND, fits, 2))
    c = to_numpy(etl.run(exe, make_key(4), N_ROUND, fits, 2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- select_rand_pbest -------------------------------------------------------


def test_pbest_shape_dtype():
    exe = build(select_rand_pbest, KEY_SPEC, 0.2, POP_SPEC, FIT_SPEC)
    out = to_numpy(etl.run(exe, make_key(0), 0.2, POP, FITNESS))
    assert out.shape == (N, 10) and out.dtype == np.float32


def test_pbest_rows_come_from_population():
    exe = build(select_rand_pbest, KEY_SPEC, 0.2, POP_SPEC, FIT_SPEC)
    out = to_numpy(etl.run(exe, make_key(0), 0.2, POP, FITNESS))
    pop_rows = {tuple(r) for r in POP}
    assert all(tuple(r) in pop_rows for r in out)


def test_pbest_rows_come_from_top_pool():
    # every selected row must be one of the top max(int(N * percent), 1) rows
    percent = 0.2
    exe = build(select_rand_pbest, KEY_SPEC, percent, POP_SPEC, FIT_SPEC)
    out = to_numpy(etl.run(exe, make_key(0), percent, POP, FITNESS))
    top_p_num = max(int(N * percent), 1)
    pool_rows = {tuple(r) for r in POP[np.argsort(FITNESS)[:top_p_num]]}
    assert all(tuple(r) in pool_rows for r in out)


def test_pbest_tiny_percent_selects_the_single_best():
    # percent so small that top_p_num clamps to 1 → every row is the best row
    percent = 0.001
    exe = build(select_rand_pbest, KEY_SPEC, percent, POP_SPEC, FIT_SPEC)
    out = to_numpy(etl.run(exe, make_key(0), percent, POP, FITNESS))
    best_row = POP[np.argmin(FITNESS)]
    np.testing.assert_array_equal(out, np.tile(best_row, (N, 1)))


def test_pbest_determinism_and_divergence():
    exe = build(select_rand_pbest, KEY_SPEC, 0.2, POP_SPEC, FIT_SPEC)
    a = to_numpy(etl.run(exe, make_key(5), 0.2, POP, FITNESS))
    b = to_numpy(etl.run(exe, make_key(5), 0.2, POP, FITNESS))
    c = to_numpy(etl.run(exe, make_key(6), 0.2, POP, FITNESS))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
