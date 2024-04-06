from __future__ import annotations

from pprint import pformat
from typing import Any, Optional, Tuple, Union
from copy import copy
import pickle
import dataclasses

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class, tree_map


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

    def _set_state_dict_mut(self, state_dict: dict) -> State:
        """Force set child state and return self

        This method mutate the struture itself and is not pure.
        Use with cautious.
        """
        self.__dict__["_state_dict"] = state_dict
        return self

    def _set_child_states_mut(self, child_states: dict) -> State:
        """Force set child state and return self

        This method mutate the struture itself and is not pure.
        Use with cautious.
        """
        self.__dict__["_child_states"] = child_states
        return self

    def _set_state_id_mut(self, state_id) -> State:
        """Force set the state id and return self

        This method mutate the struture itself and is not pure.
        Use with cautious.
        """
        self.__dict__["_state_id"] = state_id
        return self

    def update(self, **kwargs) -> State:
        """Update the current State with another State or dict and return new State.

        This method also accept keyword arguments.

        Example::
            >>> from evox import State
            >>> state = State(x=1, y=2)
            >>> state.update(y=3) # use the update method
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

    def replace(self, **kwargs) -> State:
        return self.update(**kwargs)

    def has_child(self, name: str) -> bool:
        return name in self._child_states

    def get_child_state(self, name: str) -> State:
        return self._child_states[name]

    def update_child(self, name: str, child_state: State) -> State:
        return copy(self)._set_child_states_mut(
            {**self._child_states, name: child_state}
        )

    def find_path_to(
        self, node_id: int, hint: Optional[str] = None
    ) -> Optional[Tuple[Union[Tuple, int], State]]:
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

    def update_path(self, path, new_state):
        if isinstance(path, int):
            assert path == self._state_id
            return new_state
        elif isinstance(path, tuple):
            child_id, path = path
            return self.update_child(
                child_id,
                self._child_states[child_id].update_path(path, new_state),
            )
        else:
            raise ValueError("Path must be either tuple or int")

    def __getattr__(self, key: str) -> Any:
        if is_magic_method(key):
            return super().__getattr__(key)

        if dataclasses.is_dataclass(self._state_dict):
            return getattr(self._state_dict, key)
        else:
            return self._state_dict[key]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def index(self, index: Union[str, int]) -> State:
        """
        PyTree index, apply the index to every element in the state.
        """
        return tree_map(lambda x: x[index], self)

    def slice(self, begin: int, end: int) -> State:
        """
        PyTree index, apply the index to every element in the state.
        """
        if isinstance(begin, int) and isinstance(end, int):
            return tree_map(lambda x: x[begin:end], self)
        else:
            raise TypeError(
                f"begin and end should be int, but got {type(begin)} and {type(end)}"
            )

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

    def __eq__(self, other: State):
        if self._state_dict != other._state_dict:
            return False

        return self._child_states == other._child_states

    @classmethod
    def stack(cls, states: list[State], axis=0) -> State:
        """Stack a list of states into a single state"""

        def stack_arrays(array, *araays):
            return jnp.stack((array, *araays), axis=axis)

        return tree_map(stack_arrays, states[0], states[1:])

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str) -> State:
        with open(path, "rb") as f:
            return pickle.load(f)
