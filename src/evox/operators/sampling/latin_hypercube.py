import torch


def latin_hypercube_sampling_standard(n: int, d: int, device: torch.device, smooth: bool = True):
    """Generate Latin Hypercube samples in the unit hypercube.

    :param n: The number of sample points to generate.
    :param d: The dimensionality of the samples.
    :param device: The device on which to generate the samples.
    :param smooth: Whether to generate sample in random positions in the cells or not. Defaults to True.

    :return: A tensor of shape (n, d), where each row represents a sample point and each column represents a dimension.
    """
    cells = torch.arange(0, n, device=device).view(-1, 1).expand(n, d).contiguous()
    cells_perms = torch.rand(n, d, device=device).argsort(dim=0)
    cells = cells.gather(0, cells_perms)
    if smooth:
        samples = (cells + torch.rand(n, d, device=device)) / n
    else:
        samples = (cells + 0.5) / n
    return samples


def latin_hypercube_sampling(n: int, lb: torch.Tensor, ub: torch.Tensor, smooth: bool = True):
    """Generate Latin Hypercube samples in the given hypercube defined by `lb` and `ub`.

    :param n: The number of sample points to generate.
    :param lb: The lower bounds of the hypercube. Must be a 1D tensor of size `d` with same shape, dtype, and device as `ub`.
    :param ub: The upper bounds of the hypercube. Must be a 1D tensor of size `d` with same shape, dtype, and device as `lb`.
    :param smooth: Whether to generate sample in random positions in the cells or not. Defaults to True.

    :return: A tensor of shape (n, d), where each row represents a sample point and each column represents a dimension whose device is the same as `lb` and `ub`.
    """
    assert lb.device == ub.device and lb.dtype == ub.dtype and lb.ndim == 1 and ub.ndim == 1 and lb.size() == ub.size()
    samples = latin_hypercube_sampling_standard(n, lb.size(0), lb.device, smooth)
    lb = lb[None, :]
    ub = ub[None, :]
    return lb + samples * (ub - lb)
