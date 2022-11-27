from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_leaves
import jax
import jax.numpy as jnp
import copy

import evox as ex
from evox.utils import *


@ex.jit_class
class FlattenParam:
    def __init__(self, dummy_input):
        self.shape_def = tree_map(lambda x: x.shape, dummy_input)

    def flatten(self, x):
        return tree_map(lambda x: x.reshape(x.shape[0], -1), x)

    def unflatten(self, x):
        return tree_map(lambda x, shape: x.reshape(-1, *shape), x, self.shape_def)


@ex.jit_class
class TreeAlgorithm(ex.Algorithm):
    def __init__(self, base_algorithm, initial_params, *args):
        self._base_algorithm = base_algorithm
        self.flatten_param = FlattenParam(initial_params)
        self.inner, self.treedef = tree_flatten(
            tree_map(base_algorithm, *args),
            is_leaf=lambda x: isinstance(x, ex.Algorithm),
        )

        for i, module in enumerate(self.inner):
            self.__dict__[f"auto_gen_{i}"] = module

    def ask(self, state):
        params = []
        for inner_self in self.inner:
            state, param = self._base_algorithm.ask(inner_self, state)
            params.append(param)
        params = tree_unflatten(self.treedef, params)
        return state, self.flatten_param.unflatten(params)

    def tell(self, state, fitness):
        for inner_self in self.inner:
            state = self._base_algorithm.tell(inner_self, state, fitness)

        return state
