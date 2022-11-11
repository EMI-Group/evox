import itertools
import types
from functools import partial, wraps
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .state import State


def use_state(func):
    """Decorator for easy state management.

    This decorator will try to extract the sub-state belong to the module from current state
    and merge the result back to current state.

    Parameters
    ----------
    func
        The method to be wrapped with
    """

    err_msg = "Expect first return value must be State, got {}"

    @wraps(func)
    def wrapper(self, state, *args, **kargs):
        if self.name == "_top_level" or not state.has_child(self.name):
            return_value = func(self, state, *args, **kargs)

            # single return value, the value must be a State
            if not isinstance(return_value, tuple):
                assert isinstance(return_value, State), err_msg.format(
                    type(return_value)
                )
                return state.update(return_value)

            # unpack the return value first
            assert isinstance(return_value[0], State), err_msg.format(
                type(return_value[0])
            )
            state = state.update(return_value[0])
            return (state, *return_value[1:])
        else:
            return_value = func(self, state.get_child_state(self.name), *args, **kargs)

            # single return value, the value must be a State
            if not isinstance(return_value, tuple):
                assert isinstance(return_value, State), err_msg.format(
                    type(return_value)
                )
                return state.update_child(self.name, return_value)

            # unpack the return value first
            assert isinstance(return_value[0], State), err_msg.format(
                type(return_value[0])
            )
            state = state.update_child(self.name, return_value[0])
            return (state, *return_value[1:])

    return wrapper


def jit(func):
    return jax.jit(func, static_argnums=(0,))


def jit_method(method):
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


def default_cond_fun(name):
    if name == "__call__":
        return True

    if name.startswith("_"):
        return False

    if name in ["init", "setup"]:
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
        ignore=["init", "setup"],
        ignore_prefix="_",
    ):
        wrapped = {}

        for key, value in class_dict.items():
            if key in force_wrap:
                wrapped[key] = use_state(value)
            elif key.startswith(ignore_prefix) or key in ignore:
                wrapped[key] = value
            elif callable(value):
                wrapped[key] = use_state(value)
        return super().__new__(cls, name, bases, wrapped)


class Stateful(metaclass=MetaStatefulModule):
    """Base class for all evox modules.

    This module allow easy managing of states.

    All the constants (e.g. hyperparameters) are initialized in the ``__init__``,
    and mutated states are initialized in the ``setup`` method.

    The ``init`` method will automatically call the ``setup`` of the current module
    and recursively call ``setup`` methods of all submodules.
    """

    def setup(self, key: jnp.ndarray = None) -> State:
        """Setup mutable state here

        The state it self is immutable, but it act as a mutable state
        by returning new state each time.

        Parameters
        ----------
        key : PRNGKey
            A PRNGKey.

        Returns
        -------
        State
            The state of this module.
        """
        return State()

    def init(self, key: jnp.ndarray = None, name: str = "_top_level") -> State:
        """Initialize this module and all submodules

        This method should not be overwritten.

        Parameters
        ----------
        key : PRNGKey
            A PRNGKey.
        name : string
            The name of this module.

        Returns
        -------
        State
            The state of this module and all submodules combined.
        """
        self.name = name
        child_states = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not attr_name.startswith("_") and isinstance(attr, Stateful):
                if key is None:
                    subkey = None
                else:
                    key, subkey = jax.random.split(key)
                submodule_state = attr.init(subkey, attr_name)
                assert isinstance(
                    submodule_state, State
                ), "setup method must return a State"
                child_states[attr_name] = submodule_state
        self_state = self.setup(key)
        return self_state._set_child_states(child_states)
