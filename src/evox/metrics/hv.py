import torch


def hv(objs: torch.Tensor, ref: torch.Tensor, num_sample: int = 100000):
    """
    Monte Carlo Hypervolume Calculation using bounding cube method.

    :param objs: Objective points of shape (n_points, n_objs).
    :param ref: Reference point of shape (n_objs, ).
    :param num_sample: Number of Monte Carlo samples.
    :return: Estimated hypervolume.
    """

    points = torch.abs(objs - ref)
    bound = torch.max(points, dim=0).values
    max_vol = torch.prod(bound)
    samples = torch.rand(num_sample, points.size(1)) * bound
    in_hypercube = torch.any(torch.all(samples.unsqueeze(1) < points.unsqueeze(0), dim=2), dim=1)
    hv = in_hypercube.sum() / num_sample * max_vol
    return hv
