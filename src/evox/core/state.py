import os
from pprint import pformat
from typing import Any, Optional, Tuple, Union
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

    def update(self, **kwargs) -> Self:
        warnings.warn("update() is depreacred, use replace() instead")
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

    def query_state(self, path: str) -> Self:
        """
        Recursively find a sub-state by a query name.
        eg: `'foo.bar'` will find a sub state named foo, then find `bar` under
        sub-states of `foo`
        """
        child_state = self
        for child_state_name in path.split("."):
            child_state = child_state.get_child_state(child_state_name)

        return child_state

    def replace_state(self, path: str, new_state: Self) -> Self:
        """
        replace a (sub) state by a given path
        """
        if len(path) == 0:
            return new_state

        split = path.split(".", maxsplit=1)
        if len(split) == 2:
            child_name, path = split
        else:
            child_name, path = split[0], ""

        return self.replace_child(
            child_name,
            self._child_states[child_name].replace_state(path, new_state),
        )

    def update_child(self, name: str, child_state: Self) -> Self:
        warnings.warn("update_child() is depreacred, use replace_child() instead")
        return self.replace_child(name, child_state)

    def replace_child(self, name: str, child_state: Self) -> Self:
        return copy(self)._set_child_states_mut(
            {**self._child_states, name: child_state}
        )

    def _query_state_by_id(
        self, node_id: int, module_name: Optional[str] = None
    ) -> Tuple[str, Optional[Self]]:
        """Find the state with state_id that matching the node_id

        Parameters
        -------
        node_id: int
            find the state for the module with the specified node_id
        module_name: str
            An optional module name if available
            
        Returns
        -------
        path: str
            the sub module path like `foo.bar.baz`
        state:
            the sub state with specified node_id. If not found, return None
        """
        if node_id == self._state_id:
            return "", self

        # shortcut if module_name is provided
        if module_name is not None and module_name in self._child_states:
            state = self._child_states[module_name]
            if state._state_id == node_id:
                return module_name, state

        for child_name, child_state in self._child_states.items():
            path, state = child_state._query_state_by_id(node_id, module_name)
            if state is None:
                pass
            elif len(path) > 0:
                return f"{child_name}.{path}", state
            else:
                return child_name, state

        return "", None

    def __getattr__(self, key: str) -> Any:
        if is_magic_method(key):
            return super().__getattr__(key)

        if dataclasses.is_dataclass(self._state_dict):
            return getattr(self._state_dict, key)
        else:
            return self._state_dict[key]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def index(self, index: Union[str, int]) -> Self:
        """
        PyTree index, apply the index to every element in the state.
        """
        return tree_map(lambda x: x[index], self)

    def __setattr__(self, _key: str, _value: Any) -> None:
        raise TypeError("State is immutable")

    def __setitem__(self, _key: str, _value: Any) -> None:
        raise TypeError("State is immutable")

    def __repr__(self) -> str:
        str_children = [
            f"{repr(key)}: {repr(child_state)}"
            for key, child_state in self._child_states.items()
        ]
        str_children = "{" + ",".join(str_children) + "}"
        return f"State({repr(self._state_dict)}, {str_children}, node_id: {self._state_id})"

    def __str__(self) -> str:
        return f"State({pformat(self.sprint_tree())}, node_id: {self._state_id})"

    def sprint_tree(self) -> Union[dict, str]:
        if self is State.EMPTY:
            return "State.empty"
        children = {
            key: child_state.sprint_tree()
            for key, child_state in self._child_states.items()
        }
        return self._state_dict, children

    def tree_flatten(self) -> Tuple[Tuple[dict, dict], None]:
        children = (self._state_dict, self._child_states)
        aux_data = self._state_id
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: Tuple[dict, dict]):
        state_dict, child_states = children
        state_id = aux_data
        return (
            cls()
            ._set_state_id_mut(state_id)
            ._set_state_dict_mut(state_dict)
            ._set_child_states_mut(child_states)
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
    if isinstance(obj, State):
        # TODO: check the order:
        sharding.extend(_get_state_sharding(obj._state_dict, devices))
        for child_state in obj._child_states.values():
            sharding.extend(_get_state_sharding(child_state, devices))

    elif dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            sharding.append(
                field.metadata.get("sharding", ShardingType.REPLICATED).get_sharding(
                    devices
                )
            )
    elif isinstance(obj, dict):
        # backward compatibility for dict
        # for key, value in obj.items():
        for key in obj.keys():
            sharding.append(ShardingType.REPLICATED.get_sharding(devices))

    return sharding


def get_state_sharding(state: Self, devices=None):
    flatten_sharding = _get_state_sharding(state, devices)

    return tree_unflatten(tree_structure(state), flatten_sharding)
