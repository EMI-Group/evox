from collections.abc import Iterable
from functools import partial
from typing import List, Union

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.tree_util import tree_flatten, tree_leaves, tree_unflatten

from ..core.module import *


def algorithm_has_init_ask(algorithm, state):
    probe = jax.eval_shape(algorithm.init_ask, state)
    return probe[0] is not None


def min_by(
    values: Union[jax.Array, List[jax.Array]],
    keys: Union[jax.Array, List[jax.Array]],
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


@partial(jit, static_argnames="func")
def pairwise_func(x, y, func):
    return vmap(lambda _x: vmap(lambda _y: func(_x, _y))(y))(x)


@jit
def euclidean_dist(x, y):
    return jnp.linalg.norm(x - y, axis=0)


@jit
def manhattan_dist(x, y):
    return jnp.sum(jnp.abs(x - y))


@jit
def chebyshev_dist(x, y):
    return jnp.max(jnp.abs(x - y))


@jit
def pairwise_euclidean_dist(x, y):
    return pairwise_func(x, y, euclidean_dist)


@jit
def pairwise_manhattan_dist(x, y):
    return pairwise_func(x, y, manhattan_dist)


@jit
def pairwise_chebyshev_dist(x, y):
    return pairwise_func(x, y, chebyshev_dist)


@jit
def pair_max(a, b):
    return jnp.max(a - b, axis=0)


@jit
def cos_dist(x, y):
    return jnp.dot(
        x / jnp.linalg.norm(x, axis=-1, keepdims=True),
        (y / jnp.linalg.norm(y, axis=-1, keepdims=True)).T,
    )


@jit
def cal_max(x, y):
    return pairwise_func(x, y, pair_max)


@jit
def _dominate(x, y):
    """return true if x dominate y (x < y) and false elsewise."""
    return jnp.all(x <= y) & jnp.any(x < y)


@jit
def dominate_relation(x, y):
    """return a matrix A, where A_{ij} is True if x_i donminate y_j"""
    return vmap(lambda _x: vmap(lambda _y: _dominate(_x, _y))(y))(x)


@jit
def new_dist_mat(xs: jax.Array) -> jax.Array:
    assert len(xs.shape) == 2
    xx = jax.vmap(lambda x: jnp.dot(x, x))(xs)
    x2 = jnp.broadcast_to(xx[:, jnp.newaxis], (xx.shape[0], xx.shape[0]))
    y2 = jnp.broadcast_to(xx[jnp.newaxis, :], (xx.shape[0], xx.shape[0]))
    xy = jnp.matmul(xs, xs.T)
    return jnp.sqrt(jnp.maximum(x2 + y2 - 2 * xy, 0))


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


@jit
def rank(array):
    """
    Return the rank for each item of an 1d-array.
    """
    order = jnp.argsort(array)
    rank = jnp.empty_like(order)
    rank = rank.at[order].set(jnp.arange(order.shape[0]))
    return rank


@jit
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
        return updates, state.update(opt_state=opt_state)


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


def parse_opt_direction(opt_direction: Union[str, List[str]]):
    if isinstance(opt_direction, str):
        if opt_direction == "min":
            return 1
        elif opt_direction == "max":
            return -1
        else:
            raise ValueError(
                f"opt_direction is either 'min' or 'max', got {opt_direction}"
            )
    elif isinstance(opt_direction, list):
        result = []
        for d in opt_direction:
            if opt_direction == "min":
                result.append(1)
            elif opt_direction == "max":
                result.append(-1)
            else:
                raise ValueError(f"opt_direction is either 'min' or 'max', got {d}")
        return jnp.array(result)
    else:
        raise ValueError(
            f"opt_direction should have type 'str' or 'list', got {type(opt_direction)}"
        )
