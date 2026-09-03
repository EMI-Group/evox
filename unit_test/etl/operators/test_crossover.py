"""Property tests for the functional evox_etl crossover operators (ETL only, NO torch).

Covers ``DE_differential_sum``, ``DE_binary_crossover``,
``DE_exponential_crossover``, ``simulated_binary`` and
``simulated_binary_half``. Each keyed operator is checked for determinism,
key divergence and its documented shape/dtype/contract invariants
(children-from-parents for the DE crossovers, the SBX symmetry
``c1 + c2 == p1 + p2``; the mathematically false "SBX children within parent
range" is deliberately NOT tested — beta > 1 extrapolates, same as torch).
``DE_arithmetic_recombination`` (deterministic, no key) is covered by the
torch parity suite in ``parity/``.

Note: ``DE_differential_sum``'s second output is int64 with ``replace=False``
(the select against the python-int self-pick mirror promotes the int32
randint draws) and int32 with ``replace=True`` — the tests assert the exact
observable dtype per variant.
"""

import numpy as np
import pytest

import etl
from etl import core

from evox_etl.operators.crossover import (
    DE_binary_crossover,
    DE_differential_sum,
    DE_exponential_crossover,
    simulated_binary,
    simulated_binary_half,
)

KEY_SPEC = core.TensorSpec((), np.dtype("int64"))
RNG = np.random.default_rng(0)

POP_SIZE, DIM = 32, 10
DPN = 12  # diff_padding_num

POP = RNG.standard_normal((POP_SIZE, DIM)).astype(np.float32)
POP_SPEC = core.TensorSpec((POP_SIZE, DIM), np.dtype("float32"))
INDEX = np.arange(POP_SIZE, dtype=np.int32)
INDEX_SPEC = core.TensorSpec((POP_SIZE,), np.dtype("int32"))
NDV = np.ones(POP_SIZE, dtype=np.int32)  # num_diff_vectors
NDV_SPEC = core.TensorSpec((POP_SIZE,), np.dtype("int32"))


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


# --- DE_differential_sum ------------------------------------------------------


def test_de_diff_sum_shape_dtype():
    exe = build(DE_differential_sum, KEY_SPEC, DPN, NDV_SPEC, INDEX_SPEC, POP_SPEC, None, False)
    diff_sum, first_idx = to_numpy(
        etl.run(exe, make_key(0), DPN, NDV, INDEX, POP, None, False)
    )
    assert diff_sum.shape == (POP_SIZE, DIM) and diff_sum.dtype == np.float32
    assert first_idx.shape == (POP_SIZE,) and first_idx.dtype == np.int64
    assert np.all(first_idx >= 0) and np.all(first_idx < POP_SIZE)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_de_diff_sum_no_self_pick(seed):
    # replace=False: the returned first-difference index must never equal the
    # row's own index, except for the documented pop_size-1 mirror edge case
    # (a self-pick there maps to pop_size-1 itself)
    exe = build(DE_differential_sum, KEY_SPEC, DPN, NDV_SPEC, INDEX_SPEC, POP_SPEC, None, False)
    _, first_idx = to_numpy(
        etl.run(exe, make_key(seed), DPN, NDV, INDEX, POP, None, False)
    )
    regular = INDEX != POP_SIZE - 1
    assert np.all(first_idx[regular] != INDEX[regular])


def test_de_diff_sum_f_none_equals_f_one():
    # F=None reproduces the old torch behaviour, which F=1.0 must reproduce
    # exactly for the same key
    exe_none = build(DE_differential_sum, KEY_SPEC, DPN, NDV_SPEC, INDEX_SPEC, POP_SPEC, None, False)
    exe_one = build(DE_differential_sum, KEY_SPEC, DPN, NDV_SPEC, INDEX_SPEC, POP_SPEC, 1.0, False)
    d_none, i_none = to_numpy(etl.run(exe_none, make_key(5), DPN, NDV, INDEX, POP, None, False))
    d_one, i_one = to_numpy(etl.run(exe_one, make_key(5), DPN, NDV, INDEX, POP, 1.0, False))
    assert np.array_equal(d_none, d_one)
    assert np.array_equal(i_none, i_one)


def test_de_diff_sum_scalar_f_and_0d_num_diff_vectors():
    exe = build(DE_differential_sum, KEY_SPEC, DPN, NDV_SPEC, INDEX_SPEC, POP_SPEC, 0.5, False)
    diff_sum, _ = to_numpy(etl.run(exe, make_key(0), DPN, NDV, INDEX, POP, 0.5, False))
    assert diff_sum.shape == (POP_SIZE, DIM) and diff_sum.dtype == np.float32
    # 0-d num_diff_vectors is reshaped to (1,) by the operator
    ndv0 = np.array(1, dtype=np.int32)
    exe = build(
        DE_differential_sum, KEY_SPEC, DPN, core.TensorSpec((), np.dtype("int32")),
        INDEX_SPEC, POP_SPEC, None, False,
    )
    diff_sum, _ = to_numpy(etl.run(exe, make_key(0), DPN, ndv0, INDEX, POP, None, False))
    assert diff_sum.shape == (POP_SIZE, DIM) and diff_sum.dtype == np.float32


def test_de_diff_sum_vector_f():
    f_vec = RNG.random(POP_SIZE).astype(np.float32)
    exe = build(
        DE_differential_sum, KEY_SPEC, DPN, NDV_SPEC, INDEX_SPEC, POP_SPEC,
        core.TensorSpec((POP_SIZE,), np.dtype("float32")), False,
    )
    diff_sum, _ = to_numpy(etl.run(exe, make_key(0), DPN, NDV, INDEX, POP, f_vec, False))
    assert diff_sum.shape == (POP_SIZE, DIM) and diff_sum.dtype == np.float32


def test_de_diff_sum_replace_true():
    # replace=True skips the self-pick mirror → second output stays int32
    exe = build(DE_differential_sum, KEY_SPEC, DPN, NDV_SPEC, INDEX_SPEC, POP_SPEC, None, True)
    diff_sum, first_idx = to_numpy(
        etl.run(exe, make_key(0), DPN, NDV, INDEX, POP, None, True)
    )
    assert diff_sum.shape == (POP_SIZE, DIM) and diff_sum.dtype == np.float32
    assert first_idx.shape == (POP_SIZE,) and first_idx.dtype == np.int32
    assert np.all(first_idx >= 0) and np.all(first_idx < POP_SIZE)


def test_de_diff_sum_determinism_and_divergence():
    exe = build(DE_differential_sum, KEY_SPEC, DPN, NDV_SPEC, INDEX_SPEC, POP_SPEC, None, False)
    a = to_numpy(etl.run(exe, make_key(6), DPN, NDV, INDEX, POP, None, False))
    b = to_numpy(etl.run(exe, make_key(6), DPN, NDV, INDEX, POP, None, False))
    c = to_numpy(etl.run(exe, make_key(7), DPN, NDV, INDEX, POP, None, False))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


# --- DE_binary_crossover / DE_exponential_crossover ---------------------------


@pytest.mark.parametrize("fn", [DE_binary_crossover, DE_exponential_crossover],
                         ids=["binary", "exponential"])
def test_de_crossover_children_from_parents(fn):
    mv = RNG.standard_normal((POP_SIZE, DIM)).astype(np.float32)
    cv = RNG.standard_normal((POP_SIZE, DIM)).astype(np.float32)
    exe = build(fn, KEY_SPEC, POP_SPEC, POP_SPEC, 0.5)
    out = to_numpy(etl.run(exe, make_key(0), mv, cv, 0.5))
    assert out.shape == (POP_SIZE, DIM) and out.dtype == np.float32
    assert np.isfinite(out).all()
    # each child element equals exactly one of the two parents
    assert np.all((out == mv) | (out == cv))


def test_de_binary_vector_cr():
    mv = RNG.standard_normal((POP_SIZE, DIM)).astype(np.float32)
    cv = RNG.standard_normal((POP_SIZE, DIM)).astype(np.float32)
    cr = RNG.random(POP_SIZE).astype(np.float32)
    exe = build(DE_binary_crossover, KEY_SPEC, POP_SPEC, POP_SPEC,
                core.TensorSpec((POP_SIZE,), np.dtype("float32")))
    out = to_numpy(etl.run(exe, make_key(0), mv, cv, cr))
    assert out.shape == (POP_SIZE, DIM) and out.dtype == np.float32
    assert np.all((out == mv) | (out == cv))


@pytest.mark.parametrize("fn", [DE_binary_crossover, DE_exponential_crossover],
                         ids=["binary", "exponential"])
def test_de_crossover_determinism(fn):
    mv = RNG.standard_normal((POP_SIZE, DIM)).astype(np.float32)
    cv = RNG.standard_normal((POP_SIZE, DIM)).astype(np.float32)
    exe = build(fn, KEY_SPEC, POP_SPEC, POP_SPEC, 0.5)
    a = to_numpy(etl.run(exe, make_key(1), mv, cv, 0.5))
    b = to_numpy(etl.run(exe, make_key(1), mv, cv, 0.5))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("fn", [DE_binary_crossover, DE_exponential_crossover],
                         ids=["binary", "exponential"])
def test_de_crossover_key_divergence(fn):
    mv = RNG.standard_normal((POP_SIZE, DIM)).astype(np.float32)
    cv = RNG.standard_normal((POP_SIZE, DIM)).astype(np.float32)
    exe = build(fn, KEY_SPEC, POP_SPEC, POP_SPEC, 0.5)
    a = to_numpy(etl.run(exe, make_key(1), mv, cv, 0.5))
    b = to_numpy(etl.run(exe, make_key(2), mv, cv, 0.5))
    assert not np.array_equal(a, b)


# --- simulated_binary ---------------------------------------------------------

SBX_N = 64
X = RNG.standard_normal((SBX_N, DIM)).astype(np.float32)
X_SPEC = core.TensorSpec((SBX_N, DIM), np.dtype("float32"))


def test_sbx_symmetry():
    # offspring pairs are symmetric around the parent mid: c1 + c2 == p1 + p2
    exe = build(simulated_binary, KEY_SPEC, X_SPEC, 1.0, 20.0)
    out = to_numpy(etl.run(exe, make_key(0), X, 1.0, 20.0))
    n2 = SBX_N // 2
    assert out.shape == (SBX_N, DIM) and out.dtype == np.float32
    assert np.isfinite(out).all()
    np.testing.assert_allclose(
        out[:n2] + out[n2:], X[:n2] + X[n2:], rtol=1e-6, atol=1e-5
    )


def test_sbx_pro_c_zero_copies_parents():
    # pro_c=0 forces beta to 1 → the offspring halves are the parents (swapped)
    exe = build(simulated_binary, KEY_SPEC, X_SPEC, 0.0, 20.0)
    out = to_numpy(etl.run(exe, make_key(0), X, 0.0, 20.0))
    n2 = SBX_N // 2
    np.testing.assert_allclose(out[:n2], X[:n2], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out[n2:], X[n2:], rtol=1e-6, atol=1e-6)


def test_sbx_determinism_and_divergence():
    exe = build(simulated_binary, KEY_SPEC, X_SPEC, 1.0, 20.0)
    a = to_numpy(etl.run(exe, make_key(1), X, 1.0, 20.0))
    b = to_numpy(etl.run(exe, make_key(1), X, 1.0, 20.0))
    c = to_numpy(etl.run(exe, make_key(2), X, 1.0, 20.0))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sbx_default_pro_c_dis_c_baked():
    # pro_c/dis_c omitted at build AND at run → defaults (1.0, 20.0) baked in
    exe = build(simulated_binary, KEY_SPEC, X_SPEC)
    out = to_numpy(etl.run(exe, make_key(3), X))
    ref_exe = build(simulated_binary, KEY_SPEC, X_SPEC, 1.0, 20.0)
    ref = to_numpy(etl.run(ref_exe, make_key(3), X, 1.0, 20.0))
    assert np.array_equal(out, ref)


# --- simulated_binary_half ----------------------------------------------------


def test_sbx_half_shape_dtype_finite():
    exe = build(simulated_binary_half, KEY_SPEC, X_SPEC, 1.0, 20.0)
    out = to_numpy(etl.run(exe, make_key(0), X, 1.0, 20.0))
    assert out.shape == (SBX_N // 2, DIM) and out.dtype == np.float32
    assert np.isfinite(out).all()


def test_sbx_half_determinism_and_divergence():
    exe = build(simulated_binary_half, KEY_SPEC, X_SPEC, 1.0, 20.0)
    a = to_numpy(etl.run(exe, make_key(1), X, 1.0, 20.0))
    b = to_numpy(etl.run(exe, make_key(1), X, 1.0, 20.0))
    c = to_numpy(etl.run(exe, make_key(2), X, 1.0, 20.0))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
