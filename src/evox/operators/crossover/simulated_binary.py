import jax.numpy as jnp
from jax import random, jit, vmap, lax
from evox import jit_class


def _random_pairing(key, x):
    batch, dim = x.shape
    x = random.permutation(key, x, axis=0)
    return x.reshape(batch // 2, 2, dim)


def _unpair(x):
    batch, _, dim = x.shape
    return x.reshape(batch * 2, dim)


def _sbx_crossover(key, parents, mu):
    """
    Parameters
    ----------
    mu
        The distribution factor
    """
    _, dim = parents.shape

    # obtain random_nums for each dimension from 0 to 1.0
    random_nums = random.uniform(key, shape=(dim,))
    beta = lax.cond(
        (random_nums <= 0.5).all(),
        lambda x: (2 * x) ** (1 / (1 + mu)),
        lambda x: (1 / (2 - 2 * x)) ** (1 / (1 + mu)),
        random_nums,
    )

    c1 = 0.5 * ((1 + beta) * parents[0] + (1 - beta) * parents[1])
    c2 = 0.5 * ((1 - beta) * parents[0] + (1 + beta) * parents[1])
    return jnp.stack([c1, c2])


@jit
def sbx(key, x, distribution_factor):
    pairing_key, crossover_key = random.split(key, 2)
    paired = _random_pairing(pairing_key, x)
    crossover_keys = random.split(crossover_key, paired.shape[0])
    children = vmap(_sbx_crossover)(crossover_keys, paired, distribution_factor)
    return _unpair(children)


@jit_class
class SBXCrossover:
    def __init__(self, distribution_factor=1):
        """
        Parameters
        ----------
        distribution_factor
            The beta is decided dynamically based on distribution factor
        """
        self.distribution_factor = distribution_factor

    def __call__(self, key, x):
        return sbx(key, x, self.distribution_factor)
