<<<<<<< HEAD
import jax
=======
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
import jax.numpy as jnp
from jax import random, jit, vmap, lax
from evox import jit_class

from evox import jit_class, Operator, State


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


<<<<<<< HEAD
@jit_class
class SBXCrossover(Operator):
    def __init__(self, stdvar=1.0, distribution_factor=1):
=======
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
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
        """
        Parameters
        ----------
        distribution_factor
            The beta is decided dynamically based on distribution factor
        """
        self.distribution_factor = distribution_factor

<<<<<<< HEAD
    def setup(self, key):
        return State(key=key)

    def __call__(self, state, x):
        key = state.key
        key, pairing_key, crossover_key = jax.random.split(key, 3)
        paired = _random_pairing(pairing_key, x)
        crossover_keys = jax.random.split(crossover_key, paired.shape[0])
        children = jax.vmap(_sbx_crossover)(
            crossover_keys, paired, self.distribution_factor)
        return _unpair(children), State(key=key)
=======
    def __call__(self, key, x):
        return sbx(key, x, self.distribution_factor)
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
