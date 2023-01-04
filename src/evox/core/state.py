from __future__ import annotations

from pprint import pformat
from typing import Any
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


def is_magic_method(name: str):
    return name.startswith("__") and name.endswith("__")

@register_pytree_node_class
class State:
    """A class represents state

    ``State`` is immutable, to update state, use the ``update`` method or the ``|`` operator.
    ``State`` has already implemented ``tree_flatten``, ``tree_unflatten``
    and has registered as a valid pytree node. So it can be used as pytree with JAX without any issue.
    """

    empty = {}

    def __init__(
        self, state_dict: dict = empty, child_states: dict[str, State] = empty, **kwargs
    ) -> None:
        """Construct a ``State`` from dict or keyword arguments

        Example::
            >>> import evox as ex
            >>> ex.State({"x": 1, "y": 2}) # from dict
            State ({'x': 1, 'y': 2}, [])
            >>> ex.State(x=1, y=2) # from keyword arguments
            State ({'x': 1, 'y': 2}, [])
        """
        if state_dict is State.empty:
            if child_states is not State.empty:
                raise ValueError(
                    "when using keyword argument, state_dict and child_state must be State.empty"
                )
            self.__dict__["_state_dict"] = kwargs
            self.__dict__["_child_states"] = State.empty
            return

        self.__dict__["_state_dict"] = state_dict
        self.__dict__["_child_states"] = child_states

    def update(self, other: State | dict = None, **kwargs) -> State:
        """Update the current State with another State or dict and return new State.

        This method also accept keyword arguments.

        Example::
            >>> import evox as ex
            >>> state = ex.State(x=1, y=2)
            >>> state.update(y=3) # use the update method
            State ({'x': 1, 'y': 3}, [])
            >>> state # note that State is immutable, so state isn't modified
            State ({'x': 1, 'y': 2}, [])
            >>> state | {"y": 4} # use the | operator
            State ({'x': 1, 'y': 4}, [])
        """
        if other is None:
            return State({**self._state_dict, **kwargs}, self._child_states)

        if isinstance(other, State):
            return State(
                {**self._state_dict, **other._state_dict},
                {**self._child_states, **other._child_states},
            )

        if isinstance(other, dict):
            return State({**self._state_dict, **other}, self._child_states)

        raise ValueError("other must be either State or dict")

    def has_child(self, name) -> bool:
        return name in self._child_states

    def get_child_state(self, name) -> State:
        return self._child_states[name]

    def _set_child_states(self, child_states) -> State:
        """Force set child state and return self

        This method is not pure. Only use this method when initializing modules
        """
        self.__dict__["_child_states"] = child_states
        return self

    def update_child(self, name, child_state) -> State:
        return State(
            self._state_dict,
            {**self._child_states,
                name: self._child_states[name].update(child_state)},
        )

    def __or__(self, *args, **kwargs) -> State:
        """| operator

        Same as the update method.
        """
        return self.update(*args, **kwargs)

    def __getattr__(self, key: str) -> Any:
        if is_magic_method(key):
            return super().__getattr__(key)
        return self._state_dict[key]

    def __getitem__(self, key: str) -> Any:
        return self._state_dict[key]

    def __setattr__(self, key: str, value: Any) -> None:
        raise TypeError("State is immutable")

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("State is immutable")

    def __repr__(self) -> str:
        return f"State ({self._state_dict}, {list(self._child_states.keys())})"

    def __str__(self) -> str:
        return ("State (\n"
                f" {pformat(self._state_dict)},\n"
                f" {pformat(list(self._child_states.keys()))}\n"
                ")"
                )

    def sprint_tree(self) -> str:
        if self is State.empty:
            return "State.empty"
        str_children = {key: child_state.sprint_tree() for key, child_state in self._child_states.items()}
        return ("State (\n"
                f" {pformat(self._state_dict)},\n"
                f" {pformat(str_children)}\n"
                ")"
                )

    def tree_flatten(self):
        children = (self._state_dict, self._child_states)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __eq__(self, other: State):
        if self._state_dict != other._state_dict:
            return False

        return self._child_states == other._child_states
