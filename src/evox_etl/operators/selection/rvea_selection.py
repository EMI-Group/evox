"""RVEA reference-vector guided selection (APD calculation)."""

import etl
import etl.numpy as enp

from .non_dominate import _take_along_axis


def apd_fn(x, y, z, obj, theta):
    """
    Compute the APD (Angle-Penalized Distance) based on the given inputs.

    :param x: A tensor representing the indices of the partition.
    :param y: A tensor representing the gamma.
    :param z: A tensor representing the angle.
    :param obj: A tensor of shape (n, m) representing the objectives of the solutions.
    :param theta: A tensor representing the parameter theta used for scaling the reference vector.

    :return: A tensor containing the APD values for each solution.
    """
    relu_x = etl.cast(enp.maximum(x, 0), 'int64')
    selected_z = _take_along_axis(z, relu_x, 0)
    left = (1 + obj.shape[1] * theta * selected_z) / etl.reshape(y, (1, y.shape[0]))
    norm_obj = etl.norm(obj, axis=1)
    # torch does norm_obj[x] with the RAW indices (negative indices wrap to the
    # last row, matching etl.gather's numpy-take semantics) — relu only guards
    # the selected_z gather above.
    right = etl.gather(norm_obj, x, axis=0)
    return left * right


def ref_vec_guided(x, f, v, theta):
    """
    Perform the Reference Vector Guided Evolutionary Algorithm (RVEA) selection process.

    This function selects solutions based on the Reference Vector Guided Evolutionary Algorithm.
    It calculates the distances and angles between solutions and reference vectors, and returns
    the next set of solutions to be evolved.

    :param x: A tensor of shape (n, d) representing the current population solutions.
    :param f: A tensor of shape (n, m) representing the objective values for each solution.
    :param v: A tensor of shape (r, m) representing the reference vectors.
    :param theta: A tensor representing the parameter theta used in the APD calculation.

    :return: A tuple containing:
        - next_x: The next selected solutions.
        - next_f: The objective values of the next selected solutions.
    """
    n = f.shape[0]
    nv = v.shape[0]

    obj = f - enp.min(etl.nan_to_num(f, nan=float('inf')), axis=0, keepdims=True)
    obj = etl.maximum(obj, 1e-32)

    num = etl.matmul(v, etl.transpose(v))
    den = etl.reshape(etl.norm(v, axis=1), (nv, 1)) * etl.reshape(etl.norm(v, axis=1), (1, nv))
    cosine = num / den
    cosine = etl.select(etl.eye(nv, dtype='bool'), 0.0, cosine)
    cosine = etl.clamp(cosine, 0.0, 1.0)
    gamma = enp.min(etl.acos(cosine), axis=1)

    angle = etl.acos(
        etl.clamp(
            etl.matmul(obj, etl.transpose(v))
            / (etl.reshape(etl.norm(obj, axis=1), (n, 1)) * etl.reshape(etl.norm(v, axis=1), (1, nv))),
            0.0,
            1.0,
        )
    )

    nan_mask = enp.sum(etl.cast(etl.isnan(obj), 'int32'), axis=1) > 0
    associate = etl.argmin(angle, axis=1)
    associate = etl.reshape(etl.select(nan_mask, -1, associate), (n, 1))

    IndexMatrix = etl.reshape(enp.arange(nv, dtype='int64'), (1, nv))
    partition = etl.select(
        associate == IndexMatrix,
        etl.tile(etl.reshape(enp.arange(n, dtype='int64'), (n, 1)), (1, nv)),
        -1,
    )

    mask = etl.not_equal(associate, IndexMatrix)
    mask_null = enp.sum(etl.cast(mask, 'int32'), axis=0) == n

    apd = apd_fn(partition, gamma, angle, obj, theta)
    apd = etl.select(mask, float('inf'), apd)

    next_ind = etl.argmin(apd, axis=0)
    next_x = etl.select(etl.reshape(mask_null, (nv, 1)), float('nan'), etl.gather(x, next_ind, axis=0))
    next_f = etl.select(etl.reshape(mask_null, (nv, 1)), float('nan'), etl.gather(f, next_ind, axis=0))

    return next_x, next_f
