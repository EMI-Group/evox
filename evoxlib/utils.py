import chex
from typing import Union
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_leaves, tree_unflatten

from .core.module import *
from functools import partial


def min_by(
    values: chex.Array | list[chex.Array],
    keys: chex.Array | list[chex.Array],
):
    if isinstance(values, list):
        values = jnp.concatenate(values)
        keys = jnp.concatenate(keys)

    min_index = jnp.argmin(keys)
    return values[min_index], keys[min_index]


def _prod_tuple(xs):
    prod = 1
    for x in xs:
        prod *= x

    return prod


@jax.jit
def pair_distance(a, b):
    return jnp.linalg.norm(a-b)


@jax.jit
def euclidean_dis(x, y):
    return jax.vmap(lambda _x: jax.vmap(lambda _y: pair_distance(_x, _y))(y))(x)


@jax.jit
def pair_max(a, b):
    return jnp.max(a-b)


@jax.jit
def cos_dist(x, y):
    return jnp.dot(x / jnp.linalg.norm(x, axis=-1, keepdims=True),
                   (y / jnp.linalg.norm(y, axis=-1, keepdims=True)).T)


@jax.jit
def cal_indicator(x, y):
    return jax.vmap(lambda _x: jax.vmap(lambda _y: pair_max(_x, _y))(y))(x)


@jax.jit
def cal_fitness(pop_obj, kappa):
    n = jnp.shape(pop_obj)[0]
    pop_obj = (pop_obj - jnp.tile(jnp.min(pop_obj), (n, 1))) / \
        (jnp.tile(jnp.max(pop_obj) - jnp.min(pop_obj), (n, 1)))
    I = cal_indicator(pop_obj, pop_obj)

    C = jnp.max(jnp.abs(I), axis=0)

    fitness = jnp.sum(-jnp.exp(-I / jnp.tile(C, (n, 1)) / kappa), axis=0) + 1

    return fitness, I, C


class TreeToVector:
    def __init__(self, dummy_input):
        leaves, self.treedef = tree_flatten(dummy_input)
        self.shapes = [x.shape for x in leaves]
        self.start_indices = []
        self.slice_sizes = []
        index = 0
        for shape in self.shapes:
            self.start_indices.append(index)
            size = _prod_tuple(shape)
            self.slice_sizes.append(size)
            index += size

    @partial(jax.jit, static_argnums=(0, 2))
    def to_vector(self, x, batch=False):
        leaves = tree_leaves(x)
        if batch:
            leaves = [x.reshape(x.shape[0], -1) for x in leaves]
            return jnp.concatenate(leaves, axis=1)
        else:
            leaves = [x.reshape(-1) for x in leaves]
            return jnp.concatenate(leaves, axis=0)

    @partial(jax.jit, static_argnums=(0, 2))
    def to_tree(self, x, batch=False):
        leaves = []

        if batch:
            for start_index, slice_size, shape in zip(
                self.start_indices, self.slice_sizes, self.shapes
            ):
                batch_size = x.shape[0]
                leaves.append(
                    jax.lax.dynamic_slice(
                        x, (0, start_index), (batch_size, slice_size)
                    ).reshape(batch_size, *shape)
                )
        else:
            for start_index, slice_size, shape in zip(
                self.start_indices, self.slice_sizes, self.shapes
            ):
                leaves.append(
                    jax.lax.dynamic_slice(x, (start_index,), (slice_size,)).reshape(
                        shape
                    )
                )

        return tree_unflatten(self.treedef, leaves)
