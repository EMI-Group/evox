"""Generational Distance (GD) metrics."""

import etl
import etl.numpy as enp

from ._distance import pairwise_euclidean


def gd(objs, pf):
    """
    Calculate the Generational Distance (GD) metric between a set of solutions and the Pareto front.

    :param objs: A tensor of shape (n, m), where n is the number of solutions and m is the number of objectives.
        Represents the set of solutions to be evaluated.
    :param pf: A tensor of shape (k, m), where k is the number of points on the Pareto front and m is the number
        of objectives. Represents the true Pareto front.
    :return: The GD score, a scalar representing the average distance of the solutions to the Pareto front.

    ```{tip}
    A lower GD score indicates that the approximation is closer to the Pareto front.
    ```
    """
    distances = pairwise_euclidean(objs, pf)
    min_distances = etl.min(distances, axes=1)
    score = etl.cast(etl.norm(min_distances) / float(min_distances.shape[0]), 'float32')
    return score


def gd_plus(objs, pf, p: float = 1.0):
    """
    Calculate the GD+ metric (dominance-respecting modification of GD).

    :param objs: Solutions, shape (n, m).
    :param pf: True Pareto front, shape (k, m).
    :param p: Power parameter (default 1).
    :return: The GD+ score (scalar).
    """
    # distance from each solution to the front: sqrt(sum(max(pf - obj, 0)^2))
    diff = enp.expand_dims(pf, 0) - enp.expand_dims(objs, 1)
    distances = etl.sqrt(etl.sum(etl.maximum(diff, 0.0) ** 2, axes=-1))
    min_dis = etl.min(distances, axes=1)
    return etl.cast(
        (etl.sum(min_dis ** float(p), axes=None) / float(objs.shape[0])) ** (1.0 / float(p)),
        'float32',
    )
