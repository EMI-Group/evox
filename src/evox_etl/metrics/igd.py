"""Inverted Generational Distance (IGD) metrics."""

import etl
import etl.numpy as enp

from ._distance import pairwise_euclidean


def igd(objs, pf, p: float = 1.0):
    """
    Calculate the Inverted Generational Distance (IGD) metric between a set of solutions and the Pareto front.

    :param objs: A tensor of shape (n, m), where n is the number of solutions and m is the number of objectives.
        Represents the set of solutions to be evaluated.
    :param pf: A tensor of shape (k, m), where k is the number of points on the Pareto front and m is the number
        of objectives. Represents the true Pareto front.
    :param p: The power parameter used in the calculation (default is 1).

    :return: The IGD score, a scalar representing the average distance of the solutions to the Pareto front.

    ```{tip}
    A lower IGD score indicates that the approximation is closer to the Pareto front.
    ```
    """
    min_dis = etl.min(pairwise_euclidean(pf, objs), axes=1)
    return etl.cast(
        (etl.mean(min_dis ** float(p), axes=None)) ** (1.0 / float(p)), 'float32'
    )


def igd_plus(objs, pf, p: float = 1.0):
    """
    Calculate the IGD+ metric (dominance-respecting modification of IGD).

    :param objs: Solutions, shape (n, m).
    :param pf: True Pareto front, shape (k, m).
    :param p: Power parameter (default 1).
    :return: The IGD+ score (scalar).
    """
    # distance from each front point to the solutions: sqrt(sum(max(obj - pf, 0)^2))
    diff = enp.expand_dims(objs, 0) - enp.expand_dims(pf, 1)
    distances = etl.sqrt(etl.sum(etl.maximum(diff, 0.0) ** 2, axes=-1))
    min_dis = etl.min(distances, axes=1)
    return etl.cast(
        (etl.sum(min_dis ** float(p), axes=None) / float(pf.shape[0])) ** (1.0 / float(p)),
        'float32',
    )
