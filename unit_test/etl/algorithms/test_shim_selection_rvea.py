"""Tests for the canonical RVEA selection operators (no torch).

Converted from the deprecated ``evox_etl.algorithms._shim_selection_rvea``
compat stub (which kept torch-faithful ``apd_fn``/``_cosine_similarity``
implementations); now imports the canonical ``ref_vec_guided`` and ``apd_fn``
directly. NOTE: the canonical ``apd_fn`` gathers ``norm_obj`` with ``relu(x)``
where torch does ``norm_obj[x]`` (negative indices wrap to the last row) — a
known latent divergence. The ``apd_fn`` test therefore uses non-negative
partition indices where both semantics agree; ``_cosine_similarity`` no longer
exists canonically (its math is inlined in ``ref_vec_guided`` and covered by
the hand-verified reference below).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import etl
import numpy as np

from evox_etl.operators.selection import ref_vec_guided
from evox_etl.operators.selection.rvea_selection import apd_fn


def _trace(fn, specs, static_args):
    """Build a numpy-backend executable for a plain etl function."""
    return etl.build(fn, *specs, *static_args, backend="numpy")


def _np_ref_ref_vec_guided(x, f, v, theta):
    """Plain-numpy mirror of ref_vec_guided (hand-verified reference math)."""
    n, nv = f.shape[0], v.shape[0]
    m = f.shape[1]
    obj = f - np.nanmin(f, axis=0, keepdims=True)
    obj = np.maximum(obj, 1e-32)
    vn = v / np.linalg.norm(v, axis=1, keepdims=True)
    cosine = vn @ vn.T
    np.fill_diagonal(cosine, 0.0)
    cosine = np.clip(cosine, 0.0, 1.0)
    gamma = np.min(np.arccos(cosine), axis=1)
    on = obj / np.linalg.norm(obj, axis=1, keepdims=True)
    angle = np.arccos(np.clip(on @ vn.T, 0.0, 1.0))
    nan_mask = np.isnan(obj).any(axis=1)
    associate = np.argmin(angle, axis=1)
    associate = np.where(nan_mask, -1, associate)
    partition = np.where(
        associate[:, None] == np.arange(nv)[None, :], np.arange(n)[:, None], -1
    )
    mask = associate[:, None] != np.arange(nv)[None, :]
    mask_null = mask.sum(axis=0) == n
    selected_z = angle[np.maximum(partition, 0), np.arange(nv)]
    left = (1 + m * theta * selected_z) / gamma[None, :]
    norm_obj = np.linalg.norm(obj, axis=1)
    apd = left * norm_obj[partition]
    apd = np.where(mask, np.inf, apd)
    next_ind = np.argmin(apd, axis=0)
    return (
        np.where(mask_null[:, None], np.nan, x[next_ind]),
        np.where(mask_null[:, None], np.nan, f[next_ind]),
    )


def test_apd_fn_exact_values():
    """apd_fn on tiny known tensors equals hand-computed values.

    ``x`` is chosen non-negative: the canonical ``apd_fn`` gathers ``norm_obj``
    with ``relu(x)`` while torch does ``norm_obj[x]`` (negative indices wrap to
    the last row). On this input ``relu(x) == x`` so the two agree; negative
    entries would hit the known latent divergence.
    """
    n, nv, m = 2, 2, 2
    x = np.array([[0, 0], [1, 1]], dtype=np.int64)
    y = np.array([1.0, 2.0], dtype=np.float32)
    z = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    obj = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    theta = 1.0

    # Hand math: selected_z[i, j] = z[relu(x[i, j]), j] = z[x[i, j], j] (x >= 0);
    # left = (1 + m*theta*sel)/y[j]; right = |obj|_x[i, j].
    selected_z = z[x, np.arange(nv)]
    left = (1 + m * theta * selected_z) / y[None, :]
    norm_obj = np.linalg.norm(obj, axis=1)
    expected = left * norm_obj[x]
    # spot-check one entry by pure arithmetic: (1 + 2*0.1)/1 * sqrt(5)
    assert np.isclose(expected[0, 0], 1.2 * np.sqrt(5.0))

    exe = _trace(
        apd_fn,
        [
            etl.core.TensorSpec(shape=x.shape, dtype=np.dtype("int64")),
            etl.core.TensorSpec(shape=y.shape, dtype=np.dtype("float32")),
            etl.core.TensorSpec(shape=z.shape, dtype=np.dtype("float32")),
            etl.core.TensorSpec(shape=obj.shape, dtype=np.dtype("float32")),
        ],
        [theta],
    )
    got = etl.run(exe, x, y, z, obj, theta)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_ref_vec_guided_known_selection():
    """ref_vec_guided picks the argmin-APD solution per reference vector."""
    n, d, m, nv = 4, 2, 2, 2
    x = np.array([[10.0, 20.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    f = np.array([[2.0, 1.0], [1.5, 0.5], [0.5, 2.0], [1.0, 1.0]], dtype=np.float32)
    v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    theta = 1.0

    exe = _trace(
        ref_vec_guided,
        [
            etl.core.TensorSpec(shape=x.shape, dtype=np.dtype("float32")),
            etl.core.TensorSpec(shape=f.shape, dtype=np.dtype("float32")),
            etl.core.TensorSpec(shape=v.shape, dtype=np.dtype("float32")),
        ],
        [theta],
    )
    got_x, got_f = etl.run(exe, x, f, v, theta)
    assert got_x.shape == (nv, d)
    assert got_f.shape == (nv, m)
    assert np.isfinite(got_x.numpy()).all()
    assert np.isfinite(got_f.numpy()).all()

    # hand-verified: vector 0 associates rows {0,1,3} (APD 1.6544/0.6366/1.1573 ->
    # argmin row 1), vector 1 associates row 2 (APD 0.9549 -> argmin row 2).
    exp_x = x[[1, 2]]
    exp_f = f[[1, 2]]
    np.testing.assert_allclose(got_x.numpy(), exp_x, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got_f.numpy(), exp_f, rtol=1e-5, atol=1e-6)
    # full hand-math reference must agree too
    ref_x, ref_f = _np_ref_ref_vec_guided(x, f, v, theta)
    np.testing.assert_allclose(got_x.numpy(), ref_x, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got_f.numpy(), ref_f, rtol=1e-5, atol=1e-6)


def test_ref_vec_guided_nonorthogonal_vectors():
    """ref_vec_guided with non-orthogonal reference vectors (exercises the
    canonical inlined cosine-similarity computation, which replaced the old
    shim-only ``_cosine_similarity`` helper) agrees with the hand-math reference."""
    n, d, m, nv = 4, 2, 2, 2
    x = np.array([[10.0, 20.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    f = np.array([[2.0, 1.0], [1.5, 0.5], [0.5, 2.0], [1.0, 1.0]], dtype=np.float32)
    # vector 1 sits at 45deg from vector 0 (v[0] . v[1] / |v0||v1| = 1/sqrt(2))
    v = np.array(
        [[1.0, 0.0], [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]], dtype=np.float32
    )
    theta = 1.0

    exe = _trace(
        ref_vec_guided,
        [
            etl.core.TensorSpec(shape=x.shape, dtype=np.dtype("float32")),
            etl.core.TensorSpec(shape=f.shape, dtype=np.dtype("float32")),
            etl.core.TensorSpec(shape=v.shape, dtype=np.dtype("float32")),
        ],
        [theta],
    )
    got_x, got_f = etl.run(exe, x, f, v, theta)
    assert got_x.shape == (nv, d)
    assert got_f.shape == (nv, m)
    assert np.isfinite(got_x.numpy()).all()
    assert np.isfinite(got_f.numpy()).all()
    ref_x, ref_f = _np_ref_ref_vec_guided(x, f, v, theta)
    np.testing.assert_allclose(got_x.numpy(), ref_x, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got_f.numpy(), ref_f, rtol=1e-5, atol=1e-6)


def test_ref_vec_guided_nan_solution_never_selected():
    """A solution with NaN fitness is never selected; outputs stay finite."""
    n, d, m, nv = 4, 2, 2, 2
    x = np.array([[10.0, 20.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    f = np.array([[2.0, 1.0], [1.5, 0.5], [0.5, 2.0], [1.0, 1.0]], dtype=np.float32)
    f[1, :] = np.nan
    v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    theta = 1.0

    exe = _trace(
        ref_vec_guided,
        [
            etl.core.TensorSpec(shape=x.shape, dtype=np.dtype("float32")),
            etl.core.TensorSpec(shape=f.shape, dtype=np.dtype("float32")),
            etl.core.TensorSpec(shape=v.shape, dtype=np.dtype("float32")),
        ],
        [theta],
    )
    got_x, got_f = etl.run(exe, x, f, v, theta)
    assert got_x.shape == (nv, d)
    assert got_f.shape == (nv, m)
    # no NaN rows in the output and the NaN-fitness solution is never chosen
    assert np.isfinite(got_x.numpy()).all()
    assert np.isfinite(got_f.numpy()).all()
    assert not np.allclose(got_x.numpy(), x[1], atol=0.0)
    ref_x, ref_f = _np_ref_ref_vec_guided(x, f, v, theta)
    np.testing.assert_allclose(got_x.numpy(), ref_x, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got_f.numpy(), ref_f, rtol=1e-5, atol=1e-6)
