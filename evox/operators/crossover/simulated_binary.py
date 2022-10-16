import evox as ex
import jax
import jax.numpy as jnp


def _random_pairing(key, x):
    batch, dim = x.shape
    x = jax.random.permutation(key, x, axis=0)
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
    random_nums = jax.random.uniform(key, shape=(dim,))
    beta = jax.lax.cond((random_nums <= 0.5).all(), lambda x: (2 * x) ** (1 / (1 + mu)),
                        lambda x: (1 / (2 - 2 * x)) ** (1 / (1 + mu)), random_nums)

    c1 = 0.5 * ((1 + beta) * parents[0] + (1 - beta) * parents[1])
    c2 = 0.5 * ((1 - beta) * parents[0] + (1 + beta) * parents[1])
    return jnp.stack([c1, c2])


@ex.jit_class
class SBXCrossover(ex.Operator):
    def __init__(self, stdvar=1.0, distribution_factor=1):
        """
        Parameters
        ----------
        distribution_factor
            The beta is decided dynamically based on distribution factor
        """
        self.stdvar = stdvar
        self.distribution_factor = distribution_factor

    def setup(self, key):
        return ex.State(key=key)

    def __call__(self, state, x):
        key = state.key
        key, pairing_key, crossover_key = jax.random.split(key, 3)
        paired = _random_pairing(pairing_key, x)
        crossover_keys = jax.random.split(crossover_key, paired.shape[0])
        children = jax.vmap(_sbx_crossover)(
            crossover_keys, paired, self.distribution_factor)
        return ex.State(key=key), _unpair(children)
