"""Parity tests: evox_etl selection operators vs the torch evox reference (torch allowed here).

Both sides run on the SAME numpy inputs: the etl side via ``etl.build`` +
``etl.run`` (numpy backend; etl has no eager mode, ``etl.run`` results are
converted with ``.numpy()`` — issue 16), the torch side via ``torch.tensor``
inputs. ``torch``'s ``non_dominate_rank`` must be called eagerly (never under
``torch.compile``).

Known deviations (faithful to observable torch behaviour):
- ``apd_fn``: at MASKED entries (x == -1) torch indexes ``norm_obj`` with the
  raw x (negative wrap-around → last row) while etl uses relu(x) (→ row 0).
  ``ref_vec_guided`` overwrites exactly those entries with inf, so its output
  is unaffected — the parity test for ``apd_fn`` compares the unmasked
  entries only, plus a fully-assigned (x >= 0) case where parity is total.
- ``ref_vec_guided`` returns NaN rows for reference vectors that no solution
  associates to — faithful to torch; compared with ``equal_nan=True``.
"""

import numpy as np
import pytest
import torch

import etl
from etl import core

from evox.operators.selection import (
    crowding_distance as torch_crowding_distance,
    nd_environmental_selection as torch_nd_environmental_selection,
    non_dominate_rank as torch_non_dominate_rank,
    ref_vec_guided as torch_ref_vec_guided,
)
from evox.operators.selection.non_dominate import dominate_relation as torch_dominate_relation
from evox.operators.selection.rvea_selection import apd_fn as torch_apd_fn
from evox_etl.operators.sampling import uniform_sampling
from evox_etl.operators.selection import (
    crowding_distance,
    nd_environmental_selection,
    non_dominate_rank,
    ref_vec_guided,
)
from evox_etl.operators.selection.non_dominate import dominate_relation
from evox_etl.operators.selection.rvea_selection import apd_fn

RNG = np.random.default_rng(0)

F32 = np.dtype("float32")
I64 = np.dtype("int64")
BOOL = np.dtype("bool")


def _build_run(fn, *specs, args):
    exe = etl.build(fn, *specs, backend="numpy")
    return etl.run(exe, *args)


# --- dominate_relation -------------------------------------------------------


@pytest.mark.parametrize(
    "x_shape,y_shape", [((16, 3), (16, 3)), ((16, 3), (12, 3)), ((8, 2), (10, 2))]
)
def test_dominate_relation_parity(x_shape, y_shape):
    x = RNG.standard_normal(x_shape).astype(np.float32)
    y = RNG.standard_normal(y_shape).astype(np.float32)
    out = _build_run(
        dominate_relation,
        core.TensorSpec(x_shape, F32), core.TensorSpec(y_shape, F32),
        args=(x, y),
    ).numpy()
    t_out = torch_dominate_relation(torch.tensor(x), torch.tensor(y)).numpy()
    assert out.dtype == np.bool_ == t_out.dtype
    np.testing.assert_array_equal(out, t_out)


# --- non_dominate_rank -------------------------------------------------------


@pytest.mark.parametrize("shape", [(16, 3), (32, 2)])
def test_non_dominate_rank_parity(shape):
    f = RNG.standard_normal(shape).astype(np.float32)
    out = _build_run(
        non_dominate_rank, core.TensorSpec(shape, F32), args=(f,)
    ).numpy()
    # torch side: plain eager call (never under torch.compile)
    t_out = torch_non_dominate_rank(torch.tensor(f)).numpy()
    assert out.dtype == np.int32 == t_out.dtype
    np.testing.assert_array_equal(out, t_out)


# --- crowding_distance -------------------------------------------------------


def test_crowding_distance_parity_no_mask():
    costs = RNG.standard_normal((24, 3)).astype(np.float32)
    # mask=None passed as a static arg at build AND at run (issue 17)
    out = _build_run(
        crowding_distance, core.TensorSpec((24, 3), F32), None, args=(costs, None)
    ).numpy()
    t_out = torch_crowding_distance(torch.tensor(costs), None).numpy()
    assert out.dtype == t_out.dtype == np.float32
    np.testing.assert_allclose(out, t_out, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_crowding_distance_parity_masked():
    costs = RNG.standard_normal((24, 3)).astype(np.float32)
    mask = RNG.random(24) > 0.5  # ~half True
    out = _build_run(
        crowding_distance,
        core.TensorSpec((24, 3), F32), core.TensorSpec((24,), BOOL),
        args=(costs, mask),
    ).numpy()
    t_out = torch_crowding_distance(torch.tensor(costs), torch.tensor(mask)).numpy()
    np.testing.assert_allclose(out, t_out, rtol=1e-6, atol=1e-6, equal_nan=True)


# --- nd_environmental_selection -----------------------------------------------


def test_nd_environmental_selection_parity():
    x = RNG.standard_normal((32, 5)).astype(np.float32)
    f = RNG.standard_normal((32, 2)).astype(np.float32)
    topk = 16
    outs = [
        o.numpy()
        for o in _build_run(
            nd_environmental_selection,
            core.TensorSpec((32, 5), F32), core.TensorSpec((32, 2), F32), topk,
            args=(x, f, topk),
        )
    ]
    t_outs = [
        o.numpy()
        for o in torch_nd_environmental_selection(torch.tensor(x), torch.tensor(f), topk)
    ]
    for name, a, b in zip(("x", "f", "rank", "crowding_dis"), outs, t_outs):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True,
                                   err_msg=name)


# --- apd_fn ------------------------------------------------------------------


def test_apd_fn_parity_unmasked_entries():
    # partition-like index matrix: entries are the row index (assigned) or -1
    # (masked out by ref_vec_guided). Masked entries differ from torch by
    # design (see module docstring) → compare the assigned entries only.
    n, m, r = 10, 3, 5
    obj = RNG.uniform(0.5, 2.0, (n, m)).astype(np.float32)
    z = RNG.uniform(0.0, 1.5, (n, r)).astype(np.float32)
    y = RNG.uniform(0.3, 1.5, (r,)).astype(np.float32)
    associate = RNG.integers(0, r, (n, 1)).astype(np.int64)
    x = np.where(np.arange(r)[None, :] == associate, np.arange(n)[:, None], -1).astype(np.int64)
    theta = 0.2
    out = _build_run(
        apd_fn,
        core.TensorSpec(x.shape, I64), core.TensorSpec(y.shape, F32),
        core.TensorSpec(z.shape, F32), core.TensorSpec(obj.shape, F32), theta,
        args=(x, y, z, obj, theta),
    ).numpy()
    t_out = torch_apd_fn(
        torch.tensor(x), torch.tensor(y), torch.tensor(z), torch.tensor(obj),
        torch.tensor(theta, dtype=torch.float32),
    ).numpy()
    keep = x >= 0
    np.testing.assert_allclose(out[keep], t_out[keep], rtol=1e-6, atol=1e-6)


def test_apd_fn_parity_all_entries_assigned():
    # with x fully non-negative relu(x) == x and both sides agree everywhere
    n, m, r = 8, 3, 4
    obj = RNG.uniform(0.5, 2.0, (n, m)).astype(np.float32)
    z = RNG.uniform(0.0, 1.5, (n, r)).astype(np.float32)
    y = RNG.uniform(0.3, 1.5, (r,)).astype(np.float32)
    x = RNG.integers(0, n, (n, r)).astype(np.int64)
    theta = 0.2
    out = _build_run(
        apd_fn,
        core.TensorSpec(x.shape, I64), core.TensorSpec(y.shape, F32),
        core.TensorSpec(z.shape, F32), core.TensorSpec(obj.shape, F32), theta,
        args=(x, y, z, obj, theta),
    ).numpy()
    t_out = torch_apd_fn(
        torch.tensor(x), torch.tensor(y), torch.tensor(z), torch.tensor(obj),
        torch.tensor(theta, dtype=torch.float32),
    ).numpy()
    np.testing.assert_allclose(out, t_out, rtol=1e-6, atol=1e-6)


# --- ref_vec_guided ----------------------------------------------------------


def _ref_vec_guided_both(x, f, r, theta=0.2):
    """Run the etl and torch ref_vec_guided on the same inputs; v via Das-Dennis."""
    v_exe = etl.build(uniform_sampling, r, 2, backend="numpy")
    v, _ = etl.run(v_exe, r, 2)
    v = v.numpy()
    nx, nf = (
        o.numpy()
        for o in _build_run(
            ref_vec_guided,
            core.TensorSpec(x.shape, F32), core.TensorSpec(f.shape, F32),
            core.TensorSpec(v.shape, F32), theta,
            args=(x, f, v, theta),
        )
    )
    tnx, tnf = (
        o.numpy()
        for o in torch_ref_vec_guided(
            torch.tensor(x), torch.tensor(f), torch.tensor(v),
            torch.tensor(theta, dtype=torch.float32),
        )
    )
    return nx, nf, tnx, tnf


def test_ref_vec_guided_parity_clustered_with_nan_rows():
    # tightly clustered objectives → some reference vectors associate no
    # solution → NaN rows, faithful to torch (compared with equal_nan)
    n, d = 10, 4
    f = (1.0 + 0.01 * RNG.standard_normal((n, 2))).astype(np.float32)
    x = RNG.standard_normal((n, d)).astype(np.float32)
    nx, nf, tnx, tnf = _ref_vec_guided_both(x, f, r=8)
    assert np.isnan(nx).any() and np.isnan(tnx).any()
    assert np.array_equal(np.isnan(nx), np.isnan(tnx))
    np.testing.assert_allclose(nx, tnx, rtol=1e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(nf, tnf, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_ref_vec_guided_parity_scattered():
    n, d = 10, 4
    f = RNG.uniform(0.5, 2.0, (n, 2)).astype(np.float32)
    x = RNG.standard_normal((n, d)).astype(np.float32)
    nx, nf, tnx, tnf = _ref_vec_guided_both(x, f, r=6)
    assert np.array_equal(np.isnan(nx), np.isnan(tnx))
    np.testing.assert_allclose(nx, tnx, rtol=1e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(nf, tnf, rtol=1e-6, atol=1e-6, equal_nan=True)
