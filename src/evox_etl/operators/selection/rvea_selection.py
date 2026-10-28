"""RVEA reference-vector guided selection (APD calculation)."""

import evoxc
import evoxc.numpy as enp

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
    relu_x = evoxc.cast(enp.maximum(x, 0), 'int64')
    selected_z = _take_along_axis(z, relu_x, 0)
    left = (1 + obj.shape[1] * theta * selected_z) / evoxc.reshape(y, (1, y.shape[0]))
    norm_obj = evoxc.norm(obj, axis=1)
    # torch does norm_obj[x] with the RAW indices (negative indices wrap to the
    # last row, matching evoxc.gather's numpy-take semantics) — relu only guards
    # the selected_z gather above.
    right = evoxc.gather(norm_obj, x, axis=0)
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

    obj = f - enp.min(evoxc.nan_to_num(f, nan=float('inf')), axis=0, keepdims=True)
    obj = evoxc.maximum(obj, 1e-32)

    num = evoxc.matmul(v, evoxc.transpose(v))
    den = evoxc.reshape(evoxc.norm(v, axis=1), (nv, 1)) * evoxc.reshape(evoxc.norm(v, axis=1), (1, nv))
    cosine = num / den
    cosine = evoxc.select(evoxc.eye(nv, dtype='bool'), 0.0, cosine)
    cosine = evoxc.clamp(cosine, 0.0, 1.0)
    gamma = enp.min(evoxc.acos(cosine), axis=1)

    angle = evoxc.acos(
        evoxc.clamp(
            evoxc.matmul(obj, evoxc.transpose(v))
            / (evoxc.reshape(evoxc.norm(obj, axis=1), (n, 1)) * evoxc.reshape(evoxc.norm(v, axis=1), (1, nv))),
            0.0,
            1.0,
        )
    )

    nan_mask = enp.sum(evoxc.cast(evoxc.isnan(obj), 'int32'), axis=1) > 0
    associate = evoxc.argmin(angle, axis=1)
    associate = evoxc.reshape(evoxc.select(nan_mask, -1, associate), (n, 1))

    IndexMatrix = evoxc.reshape(enp.arange(nv, dtype='int64'), (1, nv))
    partition = evoxc.select(
        associate == IndexMatrix,
        evoxc.tile(evoxc.reshape(enp.arange(n, dtype='int64'), (n, 1)), (1, nv)),
        -1,
    )

    mask = evoxc.not_equal(associate, IndexMatrix)
    mask_null = enp.sum(evoxc.cast(mask, 'int32'), axis=0) == n

    apd = apd_fn(partition, gamma, angle, obj, theta)
    apd = evoxc.select(mask, float('inf'), apd)

    next_ind = evoxc.argmin(apd, axis=0)
    next_x = evoxc.select(evoxc.reshape(mask_null, (nv, 1)), float('nan'), evoxc.gather(x, next_ind, axis=0))
    next_f = evoxc.select(evoxc.reshape(mask_null, (nv, 1)), float('nan'), evoxc.gather(f, next_ind, axis=0))

    return next_x, next_f
