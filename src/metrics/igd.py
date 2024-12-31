import torch
from ..utils import pairwise_euclidean_dist


def igd(objs: torch.Tensor, pf: torch.Tensor, p: int = 1):
    min_dis = torch.cdist(pf, objs).min(dim=1).values
    return (min_dis.pow(p).mean()).pow(1 / p)