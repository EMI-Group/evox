import os
from pprint import pformat
from typing import Any, Optional, Tuple, Union, Callable
from typing_extensions import Self
from copy import copy
from pathlib import Path
import pickle
import dataclasses

import orbax.checkpoint as ocp
import warnings
from jax.tree_util import (
    register_pytree_node_class,
    tree_map,
    tree_structure,
    tree_unflatten,
)
from .distributed import ShardingType


PathLike = Union[str, bytes, os.PathLike]


def is_magic_method(name: str):
    return name.startswith("__") and name.endswith("__")


def linkedlist_prepend(lst, item):
    return (item, lst)


def linkedlist_concat(lst1, lst2):
    """Return the concatenation of two linked lists"""
    result = lst2
    for elem in reversed(linkedlist_to_list(lst1)):
        result = linkedlist_prepend(result, elem)

    return result


def linkedlist_to_list(lst):
    """Convert a linked list to a python list"""
    result = []
    iter_lst = lst
    while iter_lst:
        first, iter_lst = iter_lst
        result.append(first)

    return result


@register_pytree_node_class
class State:
    """A class represents state

    ``State`` is immutable, to update state, use the ``update`` method or the ``|`` operator.
    ``State`` has already implemented ``tree_flatten``, ``tree_unflatten``
    and has registered as a valid pytree node. So it can be used as pytree with JAX without any issue.
    """

    EMPTY: dict = {}

    def __init__(self, _dataclass=None, /, **kwargs) -> None:
        """Construct a ``State`` from dataclass instance or keyword arguments

        Example::
            >>> from evox import State
            >>> State(x=1, y=2) # from keyword arguments
            State({'x': 1, 'y': 2}, {})
            >>> from dataclasses import dataclass
            >>> @dataclass
            >>> class Param:
            ...     x: int
            ...     y: int
            ...
            >>> param = Param(x=1, y=2)
            >>> State(param) # from dataclass instance
            State(Param(x=1, y=2), {})
        """
        if _dataclass is not None:
            assert dataclasses.is_dataclass(
                _dataclass
            ), "when using the positional argument, it must be a dataclass"
            self.__dict__["_state_dict"] = _dataclass
        else:
            self.__dict__["_state_dict"] = kwargs
        self.__dict__["_child_states"] = State.EMPTY
        self.__dict__["_state_id"] = None
        # store closures in the state
        # and restore the value separately so that they are compatible with jax's transformation
        # it's stored as a linked list to satisfy the functional programming paradigm
        self.__dict__["_callbacks"] = ()
        self.__dict__["_closure_values"] = ()

    @classmethod
    def from_dataclass(cls, dataclass) -> Self:
        """Construct a ``State`` from dataclass instance

        Example::
            >>> from evox import State
            >>> from dataclasses import dataclass
            >>> @dataclass
            >>> class Param:
            ...     x: int
            ...     y: int
            ...
            >>> param = Param(x=1, y=2)
            >>> State.from_dataclass(param)
            State(Param(x=1, y=2), {})
        """
        return cls(dataclass)

    def _set_state_dict_mut(self, state_dict: dict) -> Self:
        """Force set child state and return self

        This method mutate the struture itself and is not pure.
        Use with cautious.
        """
        self.__dict__["_state_dict"] = state_dict
        return self

    def _set_child_states_mut(self, child_states: dict) -> Self:
        """Force set child state and return self

        This method mutate the struture itself and is not pure.
        Use with cautious.
        """
        self.__dict__["_child_states"] = child_states
        return self

    def _set_state_id_mut(self, state_id) -> Self:
        """Force set the state id and return self

        This method mutate the struture itself and is not pure.
        Use with cautious.
        """
        self.__dict__["_state_id"] = state_id
        return self

    def _set_closures_mut(self, callbacks, closure_values) -> Self:
        self.__dict__["_callbacks"] = callbacks
        self.__dict__["_closure_values"] = closure_values
        return self

    def update(self, **kwargs) -> Self:
        warnings.warn(
            "update() is depreacred, use replace() instead", DeprecationWarning
        )
        return self.replace(**kwargs)

    def replace(self, **kwargs) -> Self:
        """Update the current State with another State or dict and return new State.

        This method also accept keyword arguments.

        Example::
            >>> from evox import State
            >>> state = State(x=1, y=2)
            >>> state.replace(y=3) # use the update method
            State ({'x': 1, 'y': 3}, {})
            >>> state # note that State is immutable, so state isn't modified
            State ({'x': 1, 'y': 2}, {})
        """
        if dataclasses.is_dataclass(self._state_dict):
            return copy(self)._set_state_dict_mut(
                dataclasses.replace(self._state_dict, **kwargs)
            )
        else:
            return copy(self)._set_state_dict_mut({**self._state_dict, **kwargs})

    def has_child(self, name: str) -> bool:
        return name in self._child_states

    def get_child_state(self, name: str) -> Self:
        return self._child_states[name]

    def query_state(self, name: str) -> Self:
        """
        Recursively find a sub-state by a query name.
        eg: `'foo.bar'` will find a sub state named foo, then find `bar` under
        sub-states of `foo`
        """
        child_state = self
        for child_state_name in name.split("."):
            child_state = child_state.get_child_state(child_state_name)

        return child_state

    def update_child(self, name: str, child_state: Self) -> Self:
        warnings.warn("update_child() is depreacred, use replace_child() instead")
        return self.replace_child(name, child_state)

    def replace_child(self, name: str, child_state: Self) -> Self:
        return copy(self)._set_child_states_mut(
            {**self._child_states, name: child_state}
        )

    def find_path_to(
        self, node_id: int, hint: Optional[str] = None
    ) -> Optional[Tuple[Union[Tuple, int], Self]]:
        """Find the state with node_id matching the state_id
        A hint can be given with the module_name
        """
        if node_id == self._state_id:
            return node_id, self

        if hint in self._child_states and node_id == self._child_states[hint]._state_id:
            return (hint, node_id), self._child_states[hint]

        for child_id, child_state in self._child_states.items():
            result = child_state.find_path_to(node_id)
            if result is not None:
                path, state = result
                return (child_id, path), state

        return None

    def replace_by_path(self, path, new_state):
        if isinstance(path, int):
            assert path == self._state_id
            return new_state
        elif isinstance(path, tuple):
            child_id, path = path
            return self.replace_child(
                child_id,
                self._child_states[child_id].replace_by_path(path, new_state),
            )
        else:
            raise ValueError("Path must be either tuple or int")

    def __getattr__(self, key: str) -> Any:
        if is_magic_method(key):
            return super().__getattr__(key)

        try:
            if dataclasses.is_dataclass(self._state_dict):
                return getattr(self._state_dict, key)
            else:
                return self._state_dict[key]
        except (AttributeError, KeyError) as e:
            raise KeyError(
                f"State has no attribute '{key}'."
                "This may be due to a mismatch between the state and the module. "
                "If you're trying to fit the state to a submodule, please use the `use_state` wrapper."
            ) from e

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def index(self, index: Union[str, int]) -> Self:
        """
        PyTree index, apply the index to every element in the state.
        """
        return tree_map(lambda x: x[index], self)

    def register_callback(self, callback: Callable, *args, **kwargs) -> Self:
        """
        Add a callback to the state
        """
        callbacks = (callback, self._callbacks)
        closure_values = ((args, kwargs), self._closure_values)
        return copy(self)._set_closures_mut(callbacks, closure_values)

    def clear_callbacks(self) -> Self:
        """
        Clear all the callbacks in the state
        """
        return copy(self)._set_closures_mut((), ())

    def execute_callbacks(self, clear_closures=True) -> Self:
        """
        Execute all the callbacks in the state
        """
        closures = []
        iter_callback = self._callbacks
        iter_values = self._closure_values
        while iter_callback:
            callback, iter_callback = iter_callback
            (args, kwargs), iter_values = iter_values
            closures.append((callback, args, kwargs))

        closures.reverse()
        for callback, args, kwargs in closures:
            callback(*args, **kwargs)

        if clear_closures:
            return self.clear_callbacks()
        else:
            return self

    def prepend_closure(self, other: Self) -> Self:
        """Prepend closures stored in others to the current state"""
        callbacks = linkedlist_concat(other._callbacks, self._callbacks)
        closure_values = linkedlist_concat(other._closure_values, self._closure_values)
        return copy(self)._set_closures_mut(callbacks, closure_values)

    def __setattr__(self, _key: str, _value: Any) -> None:
        raise TypeError("State is immutable")

    def __setitem__(self, _key: str, _value: Any) -> None:
        raise TypeError("State is immutable")

    def __repr__(self) -> str:
        if self is State.EMPTY:
            return "State.empty"
        str_children = [
            f"{repr(key)}: {repr(child_state)}"
            for key, child_state in self._child_states.items()
        ]
        str_children = "{" + ",".join(str_children) + "}"
        return f"State({repr(self._state_dict)}, {str_children})"

    def __str__(self) -> str:
        return f"State{pformat(self.sprint_tree())}"

    def sprint_tree(self) -> Union[dict, str]:
        if self is State.EMPTY:
            return "State.empty"
        children = {
            key: child_state.sprint_tree()
            for key, child_state in self._child_states.items()
        }
        return self._state_dict, children

    def tree_flatten(self) -> Tuple[Tuple[dict, dict], None]:
        children = (self._state_dict, self._child_states, self._closure_values)
        aux_data = (self._state_id, self._callbacks)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: Tuple[dict, dict]):
        state_dict, child_states, closure_values = children
        state_id, callbacks = aux_data
        return (
            cls()
            ._set_state_id_mut(state_id)
            ._set_state_dict_mut(state_dict)
            ._set_child_states_mut(child_states)
            ._set_closures_mut(callbacks, closure_values)
        )

    def __eq__(self, other: Self):
        # TODO: verify the correctness of the comparison
        if self._state_dict != other._state_dict:
            return False

        return self._child_states == other._child_states

    def save(self, path: PathLike, orbax: bool = True) -> None:
        """Save the state to local filesystem

        Parameters
        ----------
        path: str
            The path to save the state
        orbax: bool, default: True
            If True, use orbax to save the state, otherwise use pickle
        """
        path = Path(path).resolve()

        if orbax:
            ckpt = ocp.StandardCheckpointer()
            ckpt.save(path, args=ocp.args.StandardSave(self))
        else:
            with path.open("wb") as f:
                pickle.dump(self, f)

    def load(self, path: PathLike, orbax: bool = True) -> Self:
        """Load the saved state from disk

        Parameters
        ----------
        path: str
            The path to load the state
        orbax: bool, default: True
            If True, use orbax to load the state, otherwise use pickle
        """
        path = Path(path).resolve()
        if orbax:
            ckpt = ocp.StandardCheckpointer()
            state = ckpt.restore(path, args=ocp.args.StandardRestore(self))
        else:
            with path.open("rb") as f:
                state = pickle.load(f)

        return state


def _get_state_sharding(obj, devices=None):
    """
    Apply DFS like tree_flatten
    """
    sharding = []
    if isinstance(obj._state_dict, dict):
        # TODO: check the order:
        sharding.extend(_get_state_sharding(obj._state_dict, devices))
        for child_state in obj._child_states.values():
            sharding.extend(_get_state_sharding(child_state, devices))
    elif dataclasses.is_dataclass(obj._state_dict):
        for field in dataclasses.fields(obj._state_dict):
            sharding.append(
                field.metadata.get("sharding", ShardingType.REPLICATED).get_sharding(
                    devices
                )
            )
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")

    return sharding


def get_state_sharding(state: Self, devices=None):
    flatten_sharding = _get_state_sharding(state, devices)

    return tree_unflatten(tree_structure(state), flatten_sharding)
