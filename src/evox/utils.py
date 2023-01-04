from collections.abc import Iterable
from functools import partial
from typing import Union, List

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_leaves, tree_unflatten

from .core.module import *


def min_by(
    values: Union[jnp.ndarray, List[jnp.ndarray]],
    keys: Union[jnp.ndarray, List[jnp.ndarray]],
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
    return jnp.linalg.norm(a - b, axis=0)


@jax.jit
def euclidean_dis(x, y):
    return jax.vmap(lambda _x: jax.vmap(lambda _y: pair_distance(_x, _y))(y))(x)


@jax.jit
def pair_max(a, b):
    return jnp.max(a - b, axis=0)


@jax.jit
def cos_dist(x, y):
    return jnp.dot(
        x / jnp.linalg.norm(x, axis=-1, keepdims=True),
        (y / jnp.linalg.norm(y, axis=-1, keepdims=True)).T,
    )


@jax.jit
def cal_indicator(x, y):
    return jax.vmap(lambda _x: jax.vmap(lambda _y: pair_max(_x, _y))(y))(x)


@jax.jit
def cal_fitness(pop_obj, kappa):
    n = jnp.shape(pop_obj)[0]
    pop_obj = (pop_obj - jnp.tile(jnp.min(pop_obj), (n, 1))) / (
        jnp.tile(jnp.max(pop_obj) - jnp.min(pop_obj), (n, 1))
    )
    I = cal_indicator(pop_obj, pop_obj)

    C = jnp.max(jnp.abs(I), axis=0)

    fitness = jnp.sum(-jnp.exp(-I / jnp.tile(C, (n, 1)) / kappa), axis=0) + 1

    return fitness, I, C


def compose(*functions):
    # if argument is a single Iterable like list or tuple,
    # treat it as a list of functions
    if len(functions) == 1 and isinstance(functions[0], Iterable):
        functions = functions[0]

    def composed_function(carry):
        for function in functions:
            carry = function(carry)
        return carry

    return composed_function


@jax.jit
def rank(array):
    """
    Return the rank for each item of an 1d-array.
    """
    order = jnp.argsort(array)
    rank = jnp.empty_like(order)
    rank = rank.at[order].set(jnp.arange(order.shape[0]))
    return rank


@jax.jit
def rank_based_fitness(raw_fitness):
    num_elems = raw_fitness.shape[0]
    fitness_rank = rank(raw_fitness)
    return fitness_rank / (num_elems - 1) - 0.5


class OptaxWrapper(Stateful):
    def __init__(self, optimizer, init_params):
        self.optimizer = optimizer
        self.init_params = init_params

    def setup(self, key):
        opt_state = self.optimizer.init(self.init_params)
        return State(opt_state=opt_state)

    def update(self, state, grads, params=None):
        updates, opt_state = self.optimizer.update(grads, state.opt_state, params)
        return state.update(opt_state=opt_state), updates


@jit_class
class TreeAndVector:
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

    def to_vector(self, x):
        leaves = tree_leaves(x)
        leaves = [x.reshape(-1) for x in leaves]
        return jnp.concatenate(leaves, axis=0)

    def batched_to_vector(self, x):
        leaves = tree_leaves(x)
        leaves = [x.reshape(x.shape[0], -1) for x in leaves]
        return jnp.concatenate(leaves, axis=1)

    def to_tree(self, x):
        leaves = []
        for start_index, slice_size, shape in zip(
            self.start_indices, self.slice_sizes, self.shapes
        ):
            leaves.append(
                jax.lax.dynamic_slice(x, (start_index,), (slice_size,)).reshape(shape)
            )
        return tree_unflatten(self.treedef, leaves)

    def batched_to_tree(self, x):
        leaves = []
        for start_index, slice_size, shape in zip(
            self.start_indices, self.slice_sizes, self.shapes
        ):
            batch_size = x.shape[0]
            leaves.append(
                jax.lax.dynamic_slice(
                    x, (0, start_index), (batch_size, slice_size)
                ).reshape(batch_size, *shape)
            )
        return tree_unflatten(self.treedef, leaves)

    # need this because of the follow issue
    # TypeError: cannot pickle 'jaxlib.pytree.PyTreeDef' object
    # https://github.com/google/jax/issues/3872
    def __getstate__(self):
        dummy_tree = tree_unflatten(self.treedef, self.start_indices)
        return {
            "dummy_tree": dummy_tree,
            "shapes": self.shapes,
            "start_indices": self.start_indices,
            "slice_sizes": self.slice_sizes,
        }

    def __setstate__(self, state_dict):
        _leaves, self.treedef = tree_flatten(state_dict["dummy_tree"])
        self.shapes = state_dict["shapes"]
        self.start_indices = state_dict["start_indices"]
        self.slice_sizes = state_dict["slice_sizes"]
