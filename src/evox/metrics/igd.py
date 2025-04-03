import torch


def igd(objs: torch.Tensor, pf: torch.Tensor, p: int = 1):
    """
    Calculate the Inverted Generational Distance (IGD) metric between a set of solutions and the Pareto front.

    :param objs: A tensor of shape (n, m), where n is the number of solutions and m is the number of objectives.
                 Represents the set of solutions to be evaluated.
    :param pf: A tensor of shape (k, m), where k is the number of points on the Pareto front and m is the number
               of objectives. Represents the true Pareto front.
    :param p: The power parameter used in the calculation (default is 1). This defines the distance metric (L^p norm).

    :return: The IGD score, a scalar representing the average distance of the solutions to the Pareto front.

    :note:
        The IGD score is lower when the approximation is closer to the Pareto front.
    """
    min_dis = torch.cdist(pf, objs).min(dim=1).values
    return (min_dis.pow(p).mean()).pow(1 / p)
