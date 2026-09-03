import etl
import etl.numpy as enp
import etl.random as random


def latin_hypercube_sampling_standard(
    key: etl.Tensor, n: int, d: int, smooth: bool = True
) -> etl.Tensor:
    """Generate Latin Hypercube samples in the unit hypercube.

    :param key: The PRNG key used for sampling.
    :param n: The number of sample points to generate.
    :param d: The dimensionality of the samples.
    :param smooth: Whether to generate sample in random positions in the cells or not. Defaults to True.

    :return: A tensor of shape (n, d), where each row represents a sample point and each column represents a dimension.
    """
    cells = etl.broadcast(etl.reshape(enp.arange(n, dtype="int64"), (n, 1)), (n, d))
    k1, k2 = random.split(key)
    cells_perms = etl.argsort(random.uniform(k1, (n, d), dtype="float32"), axis=0)
    # torch `cells.gather(0, cells_perms)` is take_along_axis semantics; etl.gather
    # is numpy-take, so use the flatten trick: idx[i, j] = perms[i, j] * d + j.
    col_idx = etl.broadcast(etl.reshape(enp.arange(d, dtype="int64"), (1, d)), (n, d))
    cells = etl.reshape(
        etl.gather(etl.reshape(cells, (-1,)), etl.cast(cells_perms, "int64") * d + col_idx),
        (n, d),
    )
    cells = etl.cast(cells, "float32")
    if smooth:
        samples = (cells + random.uniform(k2, (n, d), dtype="float32")) / n
    else:
        samples = (cells + 0.5) / n
    return samples


def latin_hypercube_sampling(
    key: etl.Tensor, n: int, lb: etl.Tensor, ub: etl.Tensor, smooth: bool = True
) -> etl.Tensor:
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
    lb = etl.reshape(lb, (1, -1))
    ub = etl.reshape(ub, (1, -1))
    return lb + samples * (ub - lb)
