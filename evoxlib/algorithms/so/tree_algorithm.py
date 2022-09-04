from functools import partial, reduce
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
import jax
import jax.numpy as jnp
import copy

import evoxlib as exl
from evoxlib.utils import *

class FlattenParam:
    def __init__(self, dummy_input):
        self.shape_def = tree_map(lambda x: x.shape, dummy_input)

    def flatten(self, x):
        return tree_map(lambda x: x.reshape(x.shape[0], -1), x)


    def unflatten(self, x):
        return tree_map(lambda x, shape: x.reshape(-1, *shape), x, self.shape_def)

class TreeAlgorithm(exl.Algorithm):
    def __init__(self, base_algorithm, initial_params, *args):
        self._base_algorithm = base_algorithm
        self.flatten_param = FlattenParam(initial_params)
        self._inner = tree_map(base_algorithm, *args)

        module_list, _treedef = tree_flatten(self._inner)
        for i, module in enumerate(module_list):
            self.__dict__[f"auto_gen_{i}"] = module

    def ask(self, state):
        leaves, treedef = tree_flatten(tree_map(
            partial(self._base_algorithm.ask, state=state), self._inner
        ), is_leaf=lambda x: isinstance(x, tuple) and isinstance(x[0], exl.State))
        state, params = zip(*leaves)
        state = reduce(exl.State.update, state)
        params = tree_unflatten(treedef, params)
        return state, self.flatten_param.unflatten(params)

    def tell(self, state, x, F):
        x = self.flatten_param.flatten(x)
        def partial_tell(this, x):
            return self._base_algorithm.tell(this, state, x, F)
        
        state, _treedef = tree_flatten(tree_map(partial_tell, self._inner, x), is_leaf=lambda x: isinstance(x, exl.State))
        state = reduce(exl.State.update, state)
        return state
