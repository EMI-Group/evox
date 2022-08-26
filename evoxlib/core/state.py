from __future__ import annotations
from typing import Any

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class State:
    empty = {}

    def __init__(
        self, state_dict: dict = empty, child_states: dict[str, State] = empty, **kwargs
    ) -> None:
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

    def _get_child_state(self, name) -> State:
        return self.__dict__["_child_states"][name]

    def _set_child_states(self, child_states) -> State:
        """Force set child state and return self

        This method is not pure. Only use this method when initializing modules
        """
        self.__dict__["_child_states"] = child_states
        return self

    def _update_child(self, name, child_state) -> State:
        return State(
            self._state_dict,
            {**self._child_states, name: self._child_states[name].update(child_state)},
        )

    def __or__(self, *args, **kwargs) -> State:
        """| operator

        Same as the update method.
        """
        return self.update(*args, **kwargs)

    def __getattr__(self, key: str) -> Any:
        return self.__dict__["_state_dict"][key]

    def __getitem__(self, key: str) -> Any:
        return self.__dict__["_state_dict"][key]

    def __setattr__(self, key: str, value: Any) -> None:
        raise TypeError("State is immutable")

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("State is immutable")

    def __repr__(self) -> str:
        return f"State ({self._state_dict}, {None if self._child_states is None else list(self._child_states.keys())})"

    def tree_flatten(self):
        children = (self._state_dict, self._child_states)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
