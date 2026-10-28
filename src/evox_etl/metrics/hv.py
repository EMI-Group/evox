"""Hypervolume metrics (Monte Carlo estimators)."""

import evoxc
import evoxc.numpy as enp
import evoxc.random as random


def bounding_cube_monte_carlo_hv(key, objs, ref, num_sample: int = 100000):
    """
    Monte Carlo Hypervolume calculation using the bounding cube method.

    :param key: A PRNG key for drawing Monte Carlo samples.
    :param objs: Objective points of shape (n_points, n_objs).
    :param ref: Reference point of shape (n_objs, ).
    :param num_sample: Number of Monte Carlo samples.
    :return: Estimated hypervolume (scalar).
    """
    points = evoxc.abs(objs - ref)
    bound = evoxc.max(points, axes=0)
    max_vol = evoxc.prod(bound, axes=None)
    samples = random.uniform(key, (num_sample, points.shape[1]), 0.0, bound)
    dominated = enp.expand_dims(samples, 1) < enp.expand_dims(points, 0)
    in_all_dims = evoxc.min(dominated, axes=-1)  # all: dominated in every objective
    in_any_cube = evoxc.max(in_all_dims, axes=-1)  # any: dominated by some point
    in_hypercube = evoxc.sum(in_any_cube, axes=None)
    return evoxc.cast(in_hypercube, 'float32') / float(num_sample) * max_vol


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
    points = evoxc.abs(objs - ref)
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
        dom_count = evoxc.sum(evoxc.min(dominated, axes=-1), axes=-1)
        # each dominating hypercube gets a 1 / dom_count share of the sample
        share = evoxc.select(
            dom_count > 0, 1.0 / evoxc.maximum(evoxc.cast(dom_count, 'float32'), 1.0), 0.0
        )
        contribution = (
            evoxc.prod(points[i], axes=None)
            * evoxc.sum(share, axes=None)
            / float(num_samples_per_point)
        )
        total = total + contribution
    return evoxc.cast(total, 'float32')
