"""Functional port of the torch RVEA selection operators (rvea_selection.py).

1:1 mirror of ``src/evox/operators/selection/rvea_selection.py`` (read-only
reference) as PLAIN functions traced via ``etl.build``/``etl.run``.

Local private copies of the ``clamp_float``/``maximum``/``nanmin`` shims
(``evox_etl.algorithms._shim_utils``) are inlined until that module lands.
"""
from typing import Tuple

import etl


def _clamp_float(a: etl.SymbolicTensor, lb: float, ub: float) -> etl.SymbolicTensor:
    """Clamp `a` elementwise to [lb, ub] (torch evox clamp_float, relu-based)."""
    return a + etl.relu(lb - a) - etl.relu(a - ub)


def _maximum(a: etl.SymbolicTensor, b: float) -> etl.SymbolicTensor:
    """Elementwise maximum of `a` and scalar `b` (torch evox maximum, relu-based)."""
    return a + etl.relu(b - a)


def _nanmin(
    t: etl.SymbolicTensor, dim: int = -1, keepdim: bool = False
) -> Tuple[etl.SymbolicTensor, etl.SymbolicTensor]:
    """Min of `t` along `dim` ignoring NaN; returns (values, indices)."""
    clean = etl.select(etl.isnan(t), float("inf"), t)
    return (
        etl.min(clean, axes=dim, keepdims=keepdim),
        etl.argmin(clean, axis=dim, keepdims=keepdim),
    )


def _cosine_similarity(a: etl.SymbolicTensor, b: etl.SymbolicTensor) -> etl.SymbolicTensor:
    """Cosine similarity along the last axis (row-normalized dot product)."""
    a = a / etl.norm(a, axis=-1, keepdims=True)
    b = b / etl.norm(b, axis=-1, keepdims=True)
    return etl.sum(a * b, axes=-1)


def apd_fn(
    x: etl.SymbolicTensor,
    y: etl.SymbolicTensor,
    z: etl.SymbolicTensor,
    obj: etl.SymbolicTensor,
    theta: float,
) -> etl.SymbolicTensor:
    """Compute the APD (Angle-Penalized Distance) for each solution/vector pair."""
    n, nv = z.shape[0], z.shape[1]
    # torch.gather(z, 0, relu(x)) == z[relu(x)[i, j], j]: etl.gather indexes along
    # an axis only (out[i, j, k] = z[idx[i, j], k]), so gather from a flattened view.
    flat_idx = etl.maximum(x, 0) * nv + etl.reshape(etl.cast(etl.arange(nv), etl.int64), (1, nv))
    selected_z = etl.reshape(etl.gather(etl.reshape(z, (n * nv,)), flat_idx, axis=0), (n, nv))
    left = (1 + obj.shape[1] * theta * selected_z) / etl.reshape(y, (1, y.shape[0]))
    norm_obj = etl.norm(obj, axis=1)
    right = etl.gather(norm_obj, x, axis=0)
    return left * right


def ref_vec_guided(
    x: etl.SymbolicTensor,
    f: etl.SymbolicTensor,
    v: etl.SymbolicTensor,
    theta: float,
) -> Tuple[etl.SymbolicTensor, etl.SymbolicTensor]:
    """Select one solution per reference vector by minimal APD (RVEA selection)."""
    n = f.shape[0]
    nv = v.shape[0]
    m = f.shape[1]

    obj = f - _nanmin(f, dim=0, keepdim=True)[0]
    obj = _maximum(obj, 1e-32)

    cosine = _cosine_similarity(
        etl.reshape(v, (nv, 1, m)), etl.reshape(v, (1, nv, m))
    )
    cosine = etl.select(etl.eye(nv, dtype=etl.bool_), 0.0, cosine)
    cosine = _clamp_float(cosine, 0.0, 1.0)
    gamma = etl.min(etl.acos(cosine), axes=1)

    angle = etl.acos(
        _clamp_float(
            _cosine_similarity(etl.reshape(obj, (n, 1, m)), etl.reshape(v, (1, nv, m))),
            0.0,
            1.0,
        )
    )

    nan_mask = etl.sum(etl.isnan(obj), axes=1) > 0
    associate = etl.argmin(angle, axis=1)
    associate = etl.select(nan_mask, associate * 0 - 1, associate)
    associate = etl.reshape(associate, (n, 1))
    partition = etl.reshape(etl.arange(n), (n, 1))
    index_matrix = etl.reshape(etl.arange(nv), (1, nv))
    partition = (
        etl.equal(associate, index_matrix) * partition
        + etl.not_equal(associate, index_matrix) * -1
    )

    mask = etl.not_equal(associate, index_matrix)
    mask_null = etl.sum(mask, axes=0) == n

    apd = apd_fn(partition, gamma, angle, obj, theta)
    apd = etl.select(mask, float("inf"), apd)

    next_ind = etl.argmin(apd, axis=0)
    next_x = etl.select(
        etl.reshape(mask_null, (nv, 1)),
        float("nan"),
        etl.gather(x, next_ind, axis=0),
    )
    next_f = etl.select(
        etl.reshape(mask_null, (nv, 1)),
        float("nan"),
        etl.gather(f, next_ind, axis=0),
    )

    return next_x, next_f
