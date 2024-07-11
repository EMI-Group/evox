import dataclasses
import warnings
from functools import wraps, partial
from collections import namedtuple
from typing import Annotated, Any, Callable, Optional, Tuple, TypeVar, get_type_hints

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node, tree_map, tree_leaves


from .state import State


def use_state(func: Callable):
    """Decorator for easy state management.

    This decorator will try to extract the sub-state belong to the module from current state
    and merge the result back to current state.

    Parameters
    ----------
    func
        The method to be wrapped with
    index
        The index of a batch state to use.
        Typically used to handle batched states created from `State.batch`.
    """

    err_msg = "Expect last return value must be State, but get {}"

    def wrapper(self, state: State, *args, **kwargs):
        assert isinstance(
            state, State
        ), f"The first argument must be `State`, got {type(state)}"
        if not hasattr(self, "_node_id") or not hasattr(self, "_module_name"):
            raise ValueError(
                f"{self} is not initialized, did you forget to call `init`?"
            )

        # find the state that match the current module
        path, extracted_state = state._query_state_by_id(
            self._node_id, self._module_name
        )

        if hasattr(func, "__self__"):
            # bounded method, don't pass self
            return_value = func(extracted_state, *args, **kwargs)
        else:
            # unbounded method (class method), pass self
            return_value = func(self, extracted_state, *args, **kwargs)

        # single return value, the value must be a State
        if not isinstance(return_value, tuple):
            assert isinstance(return_value, State), err_msg.format(type(return_value))
            aux, new_extracted_state = None, return_value
            state = state.replace_state(path, new_extracted_state)
            return state
        else:
            # unpack the return value first
            assert isinstance(return_value[-1], State), err_msg.format(
                type(return_value[-1])
            )
            aux, new_extracted_state = return_value[:-1], return_value[-1]
            state = state.replace_state(path, new_extracted_state)
            return (*aux, state)

    if hasattr(func, "__self__"):
        return wraps(func)(partial(wrapper, func.__self__))
    else:
        return wraps(func)(wrapper)


def jit_cls_method(method: Callable):
    """Decorator for methods, wrapper the method with jax.jit, and set self as static argument.

    Parameters
    ----------
    method
        A python method

    Returns
    -------
    function
        A jit wrapped version of this method
    """
    return jax.jit(method, static_argnums=(0,))


def default_jit_func(name: str):
    if name == "__call__":
        return True

    if name.startswith("_"):
        return False

    return True


def jit_class(cls):
    """A helper function used to jit decorators to methods of a class

    Returns
    -------
    class
        a class with selected methods wrapped
    """
    for attr_name in dir(cls):
        func = getattr(cls, attr_name)
        if callable(func) and default_jit_func(attr_name):
            if dataclasses.is_dataclass(cls):
                wrapped = jax.jit(func)
            else:
                wrapped = jit_cls_method(func)
            setattr(cls, attr_name, wrapped)
    return cls


SubmoduleInfo = namedtuple("SubmoduleInfo", ["name", "module", "metadata"])


class Stateful:
    """Base class for all evox modules.

    This module allow easy managing of states.

    All the constants (e.g. hyperparameters) are initialized in the ``__init__``,
    and mutated states are initialized in the ``setup`` method.

    The ``init`` method will automatically call the ``setup`` of the current module
    and recursively call ``setup`` methods of all submodules.
    """

    def __init__(self):
        super().__init__()
        object.__setattr__(self, "_node_id", None)
        object.__setattr__(self, "_module_name", None)

    def setup(self, key: jax.Array) -> State:
        """Setup mutable state here

        The state it self is immutable, but it act as a mutable state
        by returning new state each time.

        Parameters
        ----------
        key
            A PRNGKey.

        Returns
        -------
        State
            The state of this module.
        """
        return State()

    def _recursive_init(
        self, key: jax.Array, node_id: int, module_name: str
    ) -> tuple[State, int]:
        # the unique id of this module, matching its state._state_id
        object.__setattr__(self, "_node_id", node_id)
        object.__setattr__(self, "_module_name", module_name)

        # preprocess and sort to make sure the order is deterministic
        # otherwise the node_id will be different across different runs
        # making save/load impossible
        submodule_infos = []
        if dataclasses.is_dataclass(self):  # TODO: use robust check
            for field in dataclasses.fields(self):
                attr = getattr(self, field.name)
                if isinstance(attr, Stateful):
                    submodule_infos.append(
                        SubmoduleInfo(field.name, attr, field.metadata)
                    )
        else:
            for attr_name in vars(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, Stateful):
                    submodule_infos.append(SubmoduleInfo(attr_name, attr, {}))

        # Find all submodules and sort them according to their name.
        # Sorting is important because it makes sure that the node_id
        # is deterministic across different runs.
        submodule_infos.sort()
        child_states = {}
        for attr_name, attr, metadata in submodule_infos:
            key, subkey = jax.random.split(key)
            submodule_state, node_id = attr._recursive_init(
                subkey, node_id + 1, attr_name
            )
            child_states[attr_name] = submodule_state

        return (
            self.setup(key)
            ._set_state_id_mut(self._node_id)
            ._set_child_states_mut(child_states),
            node_id,
        )

    def init(self, key: jax.Array) -> State:
        """Initialize this module and all submodules

        This method should not be overwritten.

        Parameters
        ----------
        key
            A PRNGKey.

        Returns
        -------
        State
            The state of this module and all submodules combined.
        """
        state, _ = self._recursive_init(key, 0, self.__class__.__name__)
        return state

    # @classmethod
    # def stack(cls, stateful_objs, axis=0):
    #     for obj in stateful_objs:
    #         assert dataclasses.is_dataclass(obj), "All objects must be dataclasses"

    #     def stack_arrays(array, *arrays):
    #         return jnp.stack((array, *arrays), axis=axis)

    #     return tree_map(stack_arrays, stateful_objs[0], *stateful_objs[1:])

    # def __len__(self) -> int:
    #     """
    #     Inspect the length of the first element in the state,
    #     usually paired with `Stateful.stack` to read the batch size
    #     """
    #     assert dataclasses.is_dataclass(self), "Length is only supported for dataclass"

    #     return len(tree_leaves(self)[0])


class StatefulWrapper(Stateful):
    """
    A wrapper class for Stateful modules.
    """

    def __init__(self, module: Stateful):
        super().__init__()
        self._module = module

    def _recursive_init(
        self, key: jax.Array, node_id: int, module_name: str
    ) -> tuple[State, int]:
        """Skip the wrapper during init"""

        # the unique id of this module, matching its state._state_id
        object.__setattr__(self, "_node_id", node_id)
        object.__setattr__(self, "_module_name", module_name)

        return self._module._recursive_init(key, node_id, module_name)

    def setup(self, key: jax.Array) -> State:
        raise NotImplementedError("This method should not be called")
