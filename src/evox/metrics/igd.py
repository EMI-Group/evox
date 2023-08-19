from jax import jit
import jax.numpy as jnp
from evox.utils import pairwise_euclidean_dist, pairwise_func
from evox import jit_class


@jit
def igd(objs, pf, p=1):
    min_dis = jnp.min(pairwise_euclidean_dist(pf, objs), axis=1)
    return (jnp.sum(min_dis**p) / pf.shape[0]) ** (1 / p)


@jit
def igd_plus_dist(pf, obj):
    return jnp.sqrt(jnp.sum(jnp.maximum(obj - pf, 0) ** 2))


@jit
def igd_plus(objs, pf, p=1):
    min_dis = jnp.min(pairwise_func(pf, objs, igd_plus_dist), axis=1)
    return (jnp.sum(min_dis**p) / pf.shape[0]) ** (1 / p)


@jit_class
class IGD:
    def __init__(self, pf, p=1):
        self.pf = pf
        self.p = p

    def __call__(self, objs):
        return igd(objs, self.pf, self.p)


@jit_class
class IGDPlus:
    def __init__(self, pf, p=1):
        self.pf = pf
        self.p = p

    def __call__(self, objs):
        return igd_plus(objs, self.pf, self.p)
