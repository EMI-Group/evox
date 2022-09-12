from functools import partial, reduce
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_leaves
import jax
import jax.numpy as jnp
import copy

import evoxlib as exl
from evoxlib.utils import *


@exl.jit_class
class FlattenParam:
    def __init__(self, dummy_input):
        self.shape_def = tree_map(lambda x: x.shape, dummy_input)

    def flatten(self, x):
        return tree_map(lambda x: x.reshape(x.shape[0], -1), x)

    def unflatten(self, x):
        return tree_map(lambda x, shape: x.reshape(-1, *shape), x, self.shape_def)


@exl.jit_class
class TreeAlgorithm(exl.Algorithm):
    def __init__(self, base_algorithm, initial_params, *args):
        self._base_algorithm = base_algorithm
        self.flatten_param = FlattenParam(initial_params)
        self.inner, self.treedef = tree_flatten(
            tree_map(base_algorithm, *args),
            is_leaf=lambda x: isinstance(x, exl.Algorithm),
        )

        for i, module in enumerate(self.inner):
            self.__dict__[f"auto_gen_{i}"] = module

    def ask(self, state):
        params = []
        for this in self.inner:
            state, param = self._base_algorithm.ask(this, state)
            params.append(param)
        params = tree_unflatten(self.treedef, params)
        return state, self.flatten_param.unflatten(params)

    def tell(self, state, xs, F):
        xs = self.flatten_param.flatten(xs)
        xs = tree_leaves(xs)

        for this, x in zip(self.inner, xs):
            state = self._base_algorithm.tell(this, state, x, F)

        return state
