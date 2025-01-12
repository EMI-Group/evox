import torch


def gd(objs: torch.Tensor, pf: torch.Tensor):
    """
    Calculate the Generational Distance (GD) metric between a set of solutions and the Pareto front.

    :param objs: A tensor of shape (n, m), where n is the number of solutions and m is the number of objectives.
                 Represents the set of solutions to be evaluated.
    :param pf: A tensor of shape (k, m), where k is the number of points on the Pareto front and m is the number
               of objectives. Represents the true Pareto front.

    :return: The GD score, a scalar representing the average distance of the solutions to the Pareto front.

    :note:
        The GD score is lower when the approximation is closer to the Pareto front.
    """
    distances = torch.cdist(objs, pf, p=2)
    min_distances = torch.min(distances, dim=1).values
    score = torch.norm(min_distances) / min_distances.size(0)
    return score
