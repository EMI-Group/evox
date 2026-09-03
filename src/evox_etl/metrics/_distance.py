"""Shared pairwise-distance helpers for the metrics package (internal)."""

import etl
import etl.numpy as enp


def pairwise_euclidean(x, y):
    """
    Euclidean distances between each row of ``x`` and each row of ``y``.

    :param x: A tensor of shape (n, m).
    :param y: A tensor of shape (k, m).
    :return: A tensor of shape (n, k) with the L2 distance between every
        pair of rows.
    """
    diff = enp.expand_dims(x, 1) - enp.expand_dims(y, 0)
    return etl.sqrt(etl.sum(diff ** 2, axes=-1))
