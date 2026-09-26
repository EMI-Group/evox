"""Hypervolume metrics (Monte Carlo estimators)."""

import etl
import etl.numpy as enp
import etl.random as random


def bounding_cube_monte_carlo_hv(key, objs, ref, num_sample: int = 100000):
    """
    Monte Carlo Hypervolume calculation using the bounding cube method.

    :param key: A PRNG key for drawing Monte Carlo samples.
    :param objs: Objective points of shape (n_points, n_objs).
    :param ref: Reference point of shape (n_objs, ).
    :param num_sample: Number of Monte Carlo samples.
    :return: Estimated hypervolume (scalar).
    """
    points = etl.abs(objs - ref)
    bound = etl.max(points, axes=0)
    max_vol = etl.prod(bound, axes=None)
    samples = random.uniform(key, (num_sample, points.shape[1]), 0.0, bound)
    dominated = enp.expand_dims(samples, 1) < enp.expand_dims(points, 0)
    in_all_dims = etl.min(dominated, axes=-1)  # all: dominated in every objective
    in_any_cube = etl.max(in_all_dims, axes=-1)  # any: dominated by some point
    in_hypercube = etl.sum(in_any_cube, axes=None)
    return etl.cast(in_hypercube, 'float32') / float(num_sample) * max_vol


def hv(key, objs, ref, num_sample: int = 100000):
    """
    Monte Carlo Hypervolume calculation using the bounding cube method.

    :param key: A PRNG key for drawing Monte Carlo samples.
    :param objs: Objective points of shape (n_points, n_objs).
    :param ref: Reference point of shape (n_objs, ).
    :param num_sample: Number of Monte Carlo samples.
    :return: Estimated hypervolume.
    """
    return bounding_cube_monte_carlo_hv(key, objs, ref, num_sample)


def each_cube_monte_carlo_hv(key, objs, ref, num_sample: int = 100000):
    """
    Monte Carlo Hypervolume using one cube per point (uniform sample per cube).

    :param key: A PRNG key for drawing Monte Carlo samples.
    :param objs: Objective points of shape (n_points, n_objs).
    :param ref: Reference point of shape (n_objs, ).
    :param num_sample: Total number of Monte Carlo samples (split evenly
        across the points).
    :return: Estimated hypervolume (scalar).
    """
    points = etl.abs(objs - ref)
    num_points = points.shape[0]
    num_samples_per_point = num_sample // num_points
    keys = random.split_n(key, num_points)
    total = 0.0
    for i in range(num_points):
        samples_i = random.uniform(
            keys[i],
            (num_samples_per_point, points.shape[1]),
            0.0,
            enp.expand_dims(points[i], 0),
        )
        dominated = enp.expand_dims(samples_i, 1) < enp.expand_dims(points, 0)
        dom_count = etl.sum(etl.min(dominated, axes=-1), axes=-1)
        # each dominating hypercube gets a 1 / dom_count share of the sample
        share = etl.select(
            dom_count > 0, 1.0 / etl.maximum(etl.cast(dom_count, 'float32'), 1.0), 0.0
        )
        contribution = (
            etl.prod(points[i], axes=None)
            * etl.sum(share, axes=None)
            / float(num_samples_per_point)
        )
        total = total + contribution
    return etl.cast(total, 'float32')
