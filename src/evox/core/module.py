from __future__ import annotations

import warnings
from copy import copy
from functools import partial, wraps
from typing import Any, Callable, Union, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node
from jax._src.tree_util import _registry  # only used in ``is_registered_pytree_node``

from .state import State


def use_state(func: Callable, allow_override=True):
    """Decorator for easy state management.

    This decorator will try to extract the sub-state belong to the module from current state
    and merge the result back to current state.

    Parameters
    ----------
    func
        The method to be wrapped with
    """

    err_msg = "Expect last return value must be State, got {}"

    @wraps(func)
    def wrapper(self, state: State, *args, **kwargs):
        assert isinstance(
            self, Stateful
        ), f"expect self to be 'Stateful', got {type(self)} {self}"
        assert isinstance(
            state, State
        ), f"expect state to be 'State', got {type(state)} {state}"
        # find the state that match the current module
        path, matched_state = state.find_path_to(self._node_id)

        return_value = func(self, matched_state, *args, **kwargs)

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

        state = state.update_path(path, new_state)

        if aux is None:
            return state
        else:
            return (*aux, state)

    return wrapper


def jit(func: Callable):
    return jax.jit(func, static_argnums=(0,))


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
    # return jax.jit(
    #     method,
    #     static_argnums=[
    #         0,
    #     ],
    # )
    return jax.jit(method)


def default_cond_fun(name: str):
    if name == "__call__":
        return True

    if name.startswith("_"):
        return False

    if name in ["init", "setup", "allow_override"]:
        return False

    return True


def _class_decorator(cls, wrapper, cond_fun=default_cond_fun):
    """A helper function used to add decorators to methods of a class

    Parameters
    ----------
    wrapper
        The decorator
    ignore
        Ignore methods in this list
    ignore_prefix
        Ignore methods with certain prefix

    Returns
    -------
    class
        a class with selected methods wrapped
    """
    for attr_name in dir(cls):
        func = getattr(cls, attr_name)
        if callable(func) and cond_fun(attr_name):
            wrapped = wrapper(func)
            setattr(cls, attr_name, wrapped)
    return cls


def jit_class(cls, cond_fun=default_cond_fun):
    return _class_decorator(cls, jit_method, cond_fun)


def is_registered_pytree_node(obj):
    return type(obj) in _registry


def is_leaf_node(obj):
    return isinstance(obj, (bool, int, float, jax.Array, np.ndarray))


class MetaStatefulModule(type):
    """Meta class used by Module

    This meta class will try to wrap methods with use_state,
    which allows easy managing of states.

    It is recommended to use a single underscore as prefix to prevent a method from being wrapped.
    Still, this behavior can be configured by passing ``force_wrap``, ``ignore`` and ``ignore_prefix``.
    """

    def __new__(
        cls,
        name,
        bases,
        class_dict,
        force_wrap=["__call__"],
        ignore=["init", "setup", "allow_override", "override"],
        ignore_prefix="_",
    ):
        wrapped = {}

        for key, value in class_dict.items():
            if key in force_wrap:
                wrapped[key] = use_state(value)
            elif key.startswith(ignore_prefix) or key in ignore:
                wrapped[key] = value
            elif callable(value):
                wrapped[key] = use_state(value, True)

        cls = super().__new__(cls, name, bases, wrapped)
        register_pytree_node(
            cls, stateful_tree_flatten, partial(stateful_tree_unflatten, cls)
        )
        return cls


def stateful_tree_flatten(self):
    var_names = []
    var_values = []
    static_var = {}

    if hasattr(self, "_cache_override"):
        cache_override = self._cache_override
    else:
        cache_override = []

    for key in vars(self):
        var = getattr(self, key)

        if key in ["_allow_override", "_node_id"]:
            static_var[key] = var
        elif key in cache_override:
            var_names.append(key)
            var_values.append(var)
        else:
            static_var[key] = var

    children = var_values
    aux = (var_names, static_var)
    return (children, aux)


def stateful_tree_unflatten(cls, aux_data, children):
    var_values = children
    var_names, static_var = aux_data

    obj = cls.__new__(cls)
    for key, value in zip(var_names, var_values):
        setattr(obj, key, value)

    obj.__dict__ = {**obj.__dict__, **static_var}

    return obj


class Stateful(metaclass=MetaStatefulModule):
    """Base class for all evox modules.

    This module allow easy managing of states.

    All the constants (e.g. hyperparameters) are initialized in the ``__init__``,
    and mutated states are initialized in the ``setup`` method.

    The ``init`` method will automatically call the ``setup`` of the current module
    and recursively call ``setup`` methods of all submodules.
    """

    def __init__(self) -> None:
        self._node_id = None
        self._cache_override = set()

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

    def _recursive_init(self, key, node_id) -> Tuple[State, int]:
        if hasattr(self, "_node_id") and self._node_id is not None:
            warnings.warn(
                "Trying to re-initialize a Stateful module that is already initialized"
            )

        self._node_id = node_id

        child_states = {}
        for attr_name in vars(self):
            attr = getattr(self, attr_name)
            if not attr_name.startswith("_") and isinstance(attr, Stateful):
                if key is None:
                    subkey = None
                else:
                    key, subkey = jax.random.split(key)
                submodule_state, node_id = attr._recursive_init(subkey, node_id + 1)
                assert isinstance(
                    submodule_state, State
                ), "setup method must return a State"
                child_states[attr_name] = submodule_state

        return (
            self.setup(key)
            ._set_state_id_mut(self._node_id)
            ._set_child_states_mut(child_states),
            node_id,
        )

    def init(self, key: jax.Array = None) -> State:
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
        state, _node_id = self._recursive_init(key, 0)
        return state

    def allow_override(self, *override_attr_names: list[str]) -> Stateful:
        """Declare certain variables to be overridable.
        The declared variables can then be used in ``override``

        This method should be called before ``init``.

        Parameters
        ----------
        override_attr_names
            A list of attribute names, for example "lambda", "phi"

        Returns
        -------
        Stateful
            Mutate this object, declare these fields to be overridable
            and return self.
        """
        assert (
            not hasattr(self, "_node_id") or self._node_id is None
        ), "allow_override can only be called before init"
        self._cache_override = override_attr_names
        return self

    def override(self, override_attrs: dict[str, Any]) -> Stateful:
        """Override certain variables by passing a dictionary
        that maps attribute names to values.

        The variables should be declared first using ``allow_override``

        Parameters
        ----------
        override_attrs
            A dict that maps attribute names to values.

        Returns
        -------
        Stateful
            A copy of self with the required fields overrided.
        """
        overrided = copy(self)
        for key, value in override_attrs.items():
            assert hasattr(
                self, key
            ), f"Attribute {key} is not overridable, to allow it, use 'allow_override' to declare."
            setattr(overrided, key, value)

        return overrided
