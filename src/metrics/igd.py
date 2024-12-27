import torch
from ..utils import pairwise_euclidean_dist


def igd(objs: torch.Tensor, pf: torch.Tensor, p: int = 1):
    min_dis = pairwise_euclidean_dist(pf, objs).min(dim=1).values
    return (min_dis.pow(p).sum() / pf.shape[0]).pow(1 / p)