import evoxc
import evoxc.numpy as enp
import evoxc.random as random


def latin_hypercube_sampling_standard(
    key: evoxc.Tensor, n: int, d: int, smooth: bool = True
) -> evoxc.Tensor:
    """Generate Latin Hypercube samples in the unit hypercube.

    :param key: The PRNG key used for sampling.
    :param n: The number of sample points to generate.
    :param d: The dimensionality of the samples.
    :param smooth: Whether to generate sample in random positions in the cells or not. Defaults to True.

    :return: A tensor of shape (n, d), where each row represents a sample point and each column represents a dimension.
    """
    cells = evoxc.broadcast(evoxc.reshape(enp.arange(n, dtype="int64"), (n, 1)), (n, d))
    k1, k2 = random.split(key)
    cells_perms = evoxc.argsort(random.uniform(k1, (n, d), dtype="float32"), axis=0)
    # torch `cells.gather(0, cells_perms)` is take_along_axis semantics; evoxc.gather
    # is numpy-take, so use the flatten trick: idx[i, j] = perms[i, j] * d + j.
    col_idx = evoxc.broadcast(evoxc.reshape(enp.arange(d, dtype="int64"), (1, d)), (n, d))
    cells = evoxc.reshape(
        evoxc.gather(evoxc.reshape(cells, (-1,)), evoxc.cast(cells_perms, "int64") * d + col_idx),
        (n, d),
    )
    cells = evoxc.cast(cells, "float32")
    if smooth:
        samples = (cells + random.uniform(k2, (n, d), dtype="float32")) / n
    else:
        samples = (cells + 0.5) / n
    return samples


def latin_hypercube_sampling(
    key: evoxc.Tensor, n: int, lb: evoxc.Tensor, ub: evoxc.Tensor, smooth: bool = True
) -> evoxc.Tensor:
    """Generate Latin Hypercube samples in the given hypercube defined by `lb` and `ub`.

    :param key: The PRNG key used for sampling.
    :param n: The number of sample points to generate.
    :param lb: The lower bounds of the hypercube. Must be a 1D tensor of size `d` with same shape and dtype as `ub`.
    :param ub: The upper bounds of the hypercube. Must be a 1D tensor of size `d` with same shape and dtype as `lb`.
    :param smooth: Whether to generate sample in random positions in the cells or not. Defaults to True.

    :return: A tensor of shape (n, d), where each row represents a sample point and each column represents a dimension.
    """
    d = lb.shape[0]
    samples = latin_hypercube_sampling_standard(key, n, d, smooth)
    lb = evoxc.reshape(lb, (1, -1))
    ub = evoxc.reshape(ub, (1, -1))
    return lb + samples * (ub - lb)
