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


def _uniform_crossover(key, parents):
    _, dim = parents.shape
    mask = jax.random.choice(key, 2, (dim,))
    c1 = jnp.where(mask, parents[0], parents[1])
    c2 = jnp.where(mask, parents[1], parents[0])
    return jnp.stack([c1, c2])


@ex.jit_class
class UniformCrossover(ex.Operator):
    def __init__(self, stdvar=1.0):
        self.stdvar = stdvar

    def setup(self, key):
        return ex.State(key=key)

    def __call__(self, state, x):
        key = state.key
        key, pairing_key, crossover_key = jax.random.split(key, 3)
        paired = _random_pairing(pairing_key, x)
        crossover_keys = jax.random.split(crossover_key, paired.shape[0])
        children = jax.vmap(_uniform_crossover)(crossover_keys, paired)
        return ex.State(key=key), _unpair(children)
