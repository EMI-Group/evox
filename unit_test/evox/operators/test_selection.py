"""Property tests for the functional evox selection operators (EvoX only, NO torch).

Only the KEYED (random) selection operators are covered here —
``tournament_selection``, ``tournament_selection_multifit`` and
``select_rand_pbest``. The deterministic ones (``dominate_relation``,
``non_dominate_rank``, ``crowding_distance``, ``nd_environmental_selection``,
``apd_fn``, ``ref_vec_guided``) are covered by the algorithm shim tests under
``unit_test/evox/algorithms/`` (with hand-verified numpy references). Each
operator is checked for determinism (same key + inputs → identical output), key
divergence, and its documented shape/dtype/bounds contract, via
``evoxc.build`` + ``evoxc.run`` on the numpy backend.
"""

import evoxc
import numpy as np
import pytest
from evoxc import core

from evox.operators.selection import (
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
    """Convert evoxc.run results (evoxc tensor or tuple thereof) to numpy/python."""
    if isinstance(out, tuple):
        return tuple(o.numpy() if hasattr(o, "numpy") else o for o in out)
    return out.numpy() if hasattr(out, "numpy") else out


def build(fn, *specs):
    """Build an executable: tensor inputs as TensorSpec, statics passed as-is."""
    return evoxc.build(fn, *specs, backend="numpy")


# --- tournament_selection ----------------------------------------------------


def test_tournament_shape_dtype_bounds():
    exe = build(tournament_selection, KEY_SPEC, N_ROUND, FIT_SPEC, 2)
    out = to_numpy(evoxc.run(exe, make_key(0), N_ROUND, FITNESS, 2))
    assert out.shape == (N_ROUND,) and out.dtype == np.int32
    assert np.all(out >= 0) and np.all(out < N)


def test_tournament_default_tournament_size_baked():
    # tournament_size omitted at build AND at run → the Python default (2) is
    # baked into the executable (issue 17) and matches the explicit variant
    exe = build(tournament_selection, KEY_SPEC, N_ROUND, FIT_SPEC)
    out = to_numpy(evoxc.run(exe, make_key(0), N_ROUND, FITNESS))
    ref_exe = build(tournament_selection, KEY_SPEC, N_ROUND, FIT_SPEC, 2)
    ref = to_numpy(evoxc.run(ref_exe, make_key(0), N_ROUND, FITNESS, 2))
    assert np.array_equal(out, ref)


def test_tournament_size_three():
    exe = build(tournament_selection, KEY_SPEC, N_ROUND, FIT_SPEC, 3)
    out = to_numpy(evoxc.run(exe, make_key(0), N_ROUND, FITNESS, 3))
    assert out.shape == (N_ROUND,) and out.dtype == np.int32
    assert np.all(out >= 0) and np.all(out < N)


def test_tournament_determinism():
    exe = build(tournament_selection, KEY_SPEC, N_ROUND, FIT_SPEC, 2)
    a = to_numpy(evoxc.run(exe, make_key(1), N_ROUND, FITNESS, 2))
    b = to_numpy(evoxc.run(exe, make_key(1), N_ROUND, FITNESS, 2))
    assert np.array_equal(a, b)


def test_tournament_key_divergence():
    exe = build(tournament_selection, KEY_SPEC, N_ROUND, FIT_SPEC, 2)
    a = to_numpy(evoxc.run(exe, make_key(1), N_ROUND, FITNESS, 2))
    b = to_numpy(evoxc.run(exe, make_key(2), N_ROUND, FITNESS, 2))
    assert not np.array_equal(a, b)


# --- 1-D output for short selections (ported fix: n_round=1 / n_round=3) ----
# Reduced rounds must not leave a spurious trailing axis: the selection index
# tensor is 1-D of shape (n_round,) for every n_round, n_round=1 included.


@pytest.mark.parametrize("n_round", [1, 3])
@pytest.mark.parametrize("tournament_size", [2, 3])
def test_tournament_short_selection_is_1d(n_round, tournament_size):
    exe = build(tournament_selection, KEY_SPEC, n_round, FIT_SPEC, tournament_size)
    out = to_numpy(evoxc.run(exe, make_key(0), n_round, FITNESS, tournament_size))
    assert out.shape == (n_round,)
    assert out.dtype == np.int32
    assert np.all(out >= 0) and np.all(out < N)


@pytest.mark.parametrize("n_round", [1, 3])
@pytest.mark.parametrize("tournament_size", [2, 3])
def test_multifit_short_selection_is_1d(n_round, tournament_size):
    fits = [FITNESS, FITNESS2]
    exe = build(
        tournament_selection_multifit, KEY_SPEC, n_round, _multifit_specs(2), tournament_size
    )
    out = to_numpy(evoxc.run(exe, make_key(0), n_round, fits, tournament_size))
    assert out.shape == (n_round,)
    assert out.dtype == np.int32
    assert np.all(out >= 0) and np.all(out < N)


# --- very large tournament: the winner is the argmin -------------------------
# FITNESS_BIG has its unique minimum away from the ends, so "argmin" cannot be
# confused with "lowest index".
FITNESS_BIG = np.abs(np.arange(N, dtype=np.float32) - 13.0)
BIG_TOURNAMENT = 10 * N  # every candidate is drawn (with duplicates) w.h.p.


def test_tournament_large_size_picks_argmin():
    exe = build(tournament_selection, KEY_SPEC, 4, FIT_SPEC, BIG_TOURNAMENT)
    out = to_numpy(evoxc.run(exe, make_key(0), 4, FITNESS_BIG, BIG_TOURNAMENT))
    assert out.shape == (4,)
    assert int(FITNESS_BIG.argmin()) == 13
    assert np.all(out == 13)


# The multifit lexicographic order follows numpy's ``lexsort``: the LAST
# fitness is the primary key, earlier ones are tie-breakers.
MULTIFIT_F0 = np.abs(np.arange(N, dtype=np.float32) - 41.0)  # argmin at 41
MULTIFIT_F1 = np.abs(np.arange(N, dtype=np.float32) - 13.0)  # argmin at 13


def test_multifit_large_size_picks_lexicographic_winner():
    fits = [MULTIFIT_F0, MULTIFIT_F1]
    exe = build(
        tournament_selection_multifit, KEY_SPEC, 4, _multifit_specs(2), BIG_TOURNAMENT
    )
    out = to_numpy(evoxc.run(exe, make_key(0), 4, fits, BIG_TOURNAMENT))
    expected = int(np.lexsort((MULTIFIT_F0, MULTIFIT_F1))[0])
    assert expected == 13  # primary key (the last one) decides
    assert out.shape == (4,)
    assert np.all(out == expected)


def test_multifit_large_size_breaks_ties_with_earlier_key():
    primary = np.full(N, 0.5, dtype=np.float32)
    primary[[3, 7]] = 0.4  # tie on the primary key between 3 and 7
    secondary = np.arange(N, dtype=np.float32)  # prefers 3 over 7
    fits = [secondary, primary]
    exe = build(
        tournament_selection_multifit, KEY_SPEC, 4, _multifit_specs(2), BIG_TOURNAMENT
    )
    out = to_numpy(evoxc.run(exe, make_key(0), 4, fits, BIG_TOURNAMENT))
    expected = int(np.lexsort((secondary, primary))[0])
    assert expected == 3
    assert out.shape == (4,)
    assert np.all(out == expected)


# --- tournament_selection_multifit -------------------------------------------


def _multifit_specs(n_fit):
    return [FIT_SPEC] * n_fit


def test_multifit_shape_dtype_bounds():
    fits = [FITNESS, FITNESS2]
    exe = build(tournament_selection_multifit, KEY_SPEC, N_ROUND, _multifit_specs(2), 2)
    out = to_numpy(evoxc.run(exe, make_key(0), N_ROUND, fits, 2))
    assert out.shape == (N_ROUND,) and out.dtype == np.int32
    assert np.all(out >= 0) and np.all(out < N)


def test_multifit_three_objectives():
    fits = [FITNESS, FITNESS2, np.linspace(1.0, 0.0, N, dtype=np.float32)]
    exe = build(tournament_selection_multifit, KEY_SPEC, N_ROUND, _multifit_specs(3), 2)
    out = to_numpy(evoxc.run(exe, make_key(0), N_ROUND, fits, 2))
    assert out.shape == (N_ROUND,) and out.dtype == np.int32
    assert np.all(out >= 0) and np.all(out < N)


def test_multifit_determinism_and_divergence():
    fits = [FITNESS, FITNESS2]
    exe = build(tournament_selection_multifit, KEY_SPEC, N_ROUND, _multifit_specs(2), 2)
    a = to_numpy(evoxc.run(exe, make_key(3), N_ROUND, fits, 2))
    b = to_numpy(evoxc.run(exe, make_key(3), N_ROUND, fits, 2))
    c = to_numpy(evoxc.run(exe, make_key(4), N_ROUND, fits, 2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- select_rand_pbest -------------------------------------------------------


def test_pbest_shape_dtype():
    exe = build(select_rand_pbest, KEY_SPEC, 0.2, POP_SPEC, FIT_SPEC)
    out = to_numpy(evoxc.run(exe, make_key(0), 0.2, POP, FITNESS))
    assert out.shape == (N, 10) and out.dtype == np.float32


def test_pbest_rows_come_from_population():
    exe = build(select_rand_pbest, KEY_SPEC, 0.2, POP_SPEC, FIT_SPEC)
    out = to_numpy(evoxc.run(exe, make_key(0), 0.2, POP, FITNESS))
    pop_rows = {tuple(r) for r in POP}
    assert all(tuple(r) in pop_rows for r in out)


def test_pbest_rows_come_from_top_pool():
    # every selected row must be one of the top max(int(N * percent), 1) rows
    percent = 0.2
    exe = build(select_rand_pbest, KEY_SPEC, percent, POP_SPEC, FIT_SPEC)
    out = to_numpy(evoxc.run(exe, make_key(0), percent, POP, FITNESS))
    top_p_num = max(int(N * percent), 1)
    pool_rows = {tuple(r) for r in POP[np.argsort(FITNESS)[:top_p_num]]}
    assert all(tuple(r) in pool_rows for r in out)


def test_pbest_tiny_percent_selects_the_single_best():
    # percent so small that top_p_num clamps to 1 → every row is the best row
    percent = 0.001
    exe = build(select_rand_pbest, KEY_SPEC, percent, POP_SPEC, FIT_SPEC)
    out = to_numpy(evoxc.run(exe, make_key(0), percent, POP, FITNESS))
    best_row = POP[np.argmin(FITNESS)]
    np.testing.assert_array_equal(out, np.tile(best_row, (N, 1)))


def test_pbest_determinism_and_divergence():
    exe = build(select_rand_pbest, KEY_SPEC, 0.2, POP_SPEC, FIT_SPEC)
    a = to_numpy(evoxc.run(exe, make_key(5), 0.2, POP, FITNESS))
    b = to_numpy(evoxc.run(exe, make_key(5), 0.2, POP, FITNESS))
    c = to_numpy(evoxc.run(exe, make_key(6), 0.2, POP, FITNESS))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
