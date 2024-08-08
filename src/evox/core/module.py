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


def use_state(func: Callable, index: int = None):
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

    err_msg = "Expect last return value must be State, got {}"

    def wrapper(self, state: State, *args, **kwargs):
        assert isinstance(
            state, State
        ), f"The first argument must be `State`, got {type(state)}"
        if not hasattr(self, "_node_id") or not hasattr(self, "_module_name"):
            raise ValueError(
                f"{self} is not initialized, did you forget to call `init`?"
            )

        # find the state that match the current module
        path, matched_state = state.find_path_to(self._node_id, self._module_name)

        if index is not None:
            extracted_state = tree_map(lambda x: x[index], matched_state)
            this_module = tree_map(lambda x: x[index], self)
        else:
            extracted_state = matched_state
            this_module = self

        if hasattr(func, "__self__"):
            # bounded method, don't pass self
            return_value = func(extracted_state, *args, **kwargs)
        else:
            # unbounded method (class method), pass self
            return_value = func(this_module, extracted_state, *args, **kwargs)

        # single return value, the value must be a State
        if not isinstance(return_value, tuple):
            assert isinstance(return_value, State), err_msg.format(type(return_value))
            aux, new_state = None, return_value
        else:
            # unpack the return value first
            assert isinstance(return_value[-1], State), err_msg.format(
                type(return_value[-1])
            )
            aux, new_state = return_value[:-1], return_value[-1]

        # if index is specified, apply the index to the state
        if index is not None:
            new_state = tree_map(
                lambda batch_arr, new_arr: batch_arr.at[index].set(new_arr),
                matched_state,
                new_state,
            )

        state = state.replace_by_path(path, new_state).prepend_closure(new_state)

        if aux is None:
            return state
        else:
            return (*aux, state)

    if hasattr(func, "__self__"):
        return wraps(func)(partial(wrapper, func.__self__))
    else:
        return wraps(func)(wrapper)


def jit_method(method: Callable):
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
    return jax.jit(
        method,
        static_argnums=[
            0,
        ],
    )


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
                wrapped = jit_method(func)
            setattr(cls, attr_name, wrapped)
    return cls


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
        self, key: jax.Array, node_id: int, module_name: str, no_state: bool
    ) -> Tuple[State, int]:
        object.__setattr__(self, "_node_id", node_id)
        object.__setattr__(self, "_module_name", module_name)

        if not no_state:
            child_states = {}

        # Find all submodules and sort them according to their name.
        # Sorting is important because it makes sure that the node_id
        # is deterministic across different runs.
        SubmoduleInfo = namedtuple("Submodule", ["name", "module", "metadata"])

        submodules = []
        # preprocess and sort to make sure the order is deterministic
        # otherwise the node_id will be different across different runs
        # making save/load impossible
        if dataclasses.is_dataclass(self):
            for field in dataclasses.fields(self):
                attr = getattr(self, field.name)

                if isinstance(attr, Stateful):
                    submodules.append(SubmoduleInfo(field.name, attr, field.metadata))
        else:
            for attr_name in vars(self):
                attr = getattr(self, attr_name)
                if not attr_name.startswith("_") and isinstance(attr, Stateful):
                    submodules.append(SubmoduleInfo(attr_name, attr, {}))

        submodules.sort()

        for attr_name, attr, metadata in submodules:
            if key is None:
                subkey = None
            else:
                key, subkey = jax.random.split(key)

            # handle "StackAnnotation"
            # attr should be a list, or tuple of modules
            if metadata.get("stack", False):
                num_copies = len(attr)
                subkeys = jax.random.split(subkey, num_copies)
                current_node_id = node_id
                _, node_id = attr._recursive_init(None, node_id + 1, attr_name, True)
                submodule_state, _node_id = jax.vmap(
                    partial(
                        Stateful._recursive_init,
                        node_id=current_node_id + 1,
                        module_name=attr_name,
                        no_state=no_state,
                    )
                )(attr, subkeys)
            else:
                submodule_state, node_id = attr._recursive_init(
                    subkey, node_id + 1, attr_name, no_state
                )

            if not no_state:
                assert isinstance(
                    submodule_state, State
                ), "setup method must return a State"
                child_states[attr_name] = submodule_state
        if no_state:
            return None, node_id
        else:
            self_state = self.setup(key)
            if dataclasses.is_dataclass(self_state):
                # if the setup method return a dataclass, convert it to State first
                self_state = State.from_dataclass(self_state)

            self_state._set_state_id_mut(self._node_id)._set_child_states_mut(
                child_states
            )
            return self_state, node_id

    def init(self, key: jax.Array = None, no_state: bool = False) -> State:
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
        state, _node_id = self._recursive_init(key, 0, None, no_state)
        return state

    @classmethod
    def stack(cls, stateful_objs, axis=0):
        for obj in stateful_objs:
            assert dataclasses.is_dataclass(obj), "All objects must be dataclasses"

        def stack_arrays(array, *arrays):
            return jnp.stack((array, *arrays), axis=axis)

        return tree_map(stack_arrays, stateful_objs[0], *stateful_objs[1:])

    def __len__(self) -> int:
        """
        Inspect the length of the first element in the state,
        usually paired with `Stateful.stack` to read the batch size
        """
        assert dataclasses.is_dataclass(self), "Length is only supported for dataclass"

        return len(tree_leaves(self)[0])
