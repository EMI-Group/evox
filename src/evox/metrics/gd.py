from jax import jit
import jax.numpy as jnp
from evox.utils import pairwise_euclidean_dist, pairwise_func
from evox import jit_class


@jit
def gd(objs, pf, p=1):
    min_dis = jnp.min(pairwise_euclidean_dist(objs, pf), axis=1)
    return (jnp.sum(min_dis**p) / objs.shape[0]) ** (1 / p)


@jit
def gd_plus_dist(pf, obj):
    return jnp.sqrt(jnp.sum(jnp.maximum(pf - obj, 0) ** 2))


@jit
def gd_plus(objs, pf, p=1):
    min_dis = jnp.min(pairwise_func(objs, pf, gd_plus_dist), axis=1)
    return (jnp.sum(min_dis**p) / objs.shape[0]) ** (1 / p)


@jit_class
class GD:
    def __init__(self, pf, p=1):
        self.pf = pf
        self.p = p

    def __call__(self, objs):
        return gd(objs, self.pf, self.p)


@jit_class
class GDPlus:
    def __init__(self, pf, p=1):
        self.pf = pf
        self.p = p

    def __call__(self, objs):
        return gd_plus(objs, self.pf, self.p)
