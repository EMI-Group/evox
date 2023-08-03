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


def _uniform_crossover(key, parents):
    _, dim = parents.shape
    mask = random.choice(key, 2, (dim,))
    c1 = jnp.where(mask, parents[0], parents[1])
    c2 = jnp.where(mask, parents[1], parents[0])
    return jnp.stack([c1, c2])


@jit
def uniform_rand(key, x):
    key, pairing_key, crossover_key = random.split(key, 3)
    paired = _random_pairing(pairing_key, x)
    crossover_keys = random.split(crossover_key, paired.shape[0])
    children = vmap(_uniform_crossover)(crossover_keys, paired)
    return _unpair(children)


@jit_class
class UniformRand:
    def __call__(self, key, x):
        return uniform_rand(key, x)
