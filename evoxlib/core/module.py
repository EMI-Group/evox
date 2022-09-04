import chex
import jax
import types
from functools import partial, wraps

from .state import State


def jit(func):
    return jax.jit(func, static_argnums=(0,))


def use_state(func):
    """Decorator for easy state management.

    This decorator will try to extract the sub-state belong to the module from current state
    and merge the result back to current state.

    Parameters
    ----------
    func
        The method to be wrapped with
    """

    @wraps(func)
    def wrapper(self, state, *args, **kargs):
        if self.name == "_top_level":
            return_value = func(self, state, *args, **kargs)

            # single return value, the value must be a State
            if not isinstance(return_value, tuple):
                return state.update(return_value)

            # unpack the return value first
            state = state.update(return_value[0])
            return state, *return_value[1:]
        else:
            return_value = func(self, state._get_child_state(self.name), *args, **kargs)

            # single return value, the value must be a State
            if not isinstance(return_value, tuple):
                return state._update_child(self.name, return_value)

            # unpack the return value first
            state = state._update_child(self.name, return_value[0])
            return state, *return_value[1:]

    return wrapper


def vmap_method(method):
    """wrap vmap over normal methods."""

    def wrapped(self, *args, **kargs):
        return jax.vmap(partial(method, self))(*args, **kargs)

    return wrapped


def vmap_setup(setup_method, n):
    """wrap setup method.

    It's different from vmap_method in that it will automatically split the RNG key.
    """

    def wrapped(self, key):
        keys = jax.random.split(key, n)
        return jax.vmap(partial(setup_method, self))(keys)

    return wrapped


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


def _class_decorator(cls, wrapper, ignore, ignore_prefix):
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
        if attr_name.startswith(ignore_prefix):
            continue
        if attr_name in ignore:
            continue

        attr = getattr(cls, attr_name)
        if isinstance(attr, types.FunctionType):
            wrapped = wrapper(attr)
            setattr(cls, attr_name, wrapped)
    return cls


def vmap_class(cls, n, ignore=["init", "__init__"], ignore_prefix="_"):
    class VmapWrapped(cls):
        pass

    for attr_name in dir(VmapWrapped):
        if attr_name.startswith(ignore_prefix):
            continue
        if attr_name in ignore:
            continue

        attr = getattr(cls, attr_name)
        if attr_name == "setup":
            wrapped = vmap_setup(attr, n)
        else:
            wrapped = vmap_method(attr)
        setattr(VmapWrapped, attr_name, wrapped)
    return VmapWrapped


def jit_class(cls, ignore=["init", "__init__"], ignore_prefix="_"):
    return _class_decorator(cls, jit_method, ignore, ignore_prefix)


class MetaModule(type):
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


class Module(metaclass=MetaModule):
    """Base class for all EvoXLib modules.

    This module allow easy managing of states.

    All the constants (e.g. hyperparameters) are initialized in the ``__init__``,
    and mutated states are initialized in the ``setup`` method.

    The ``init`` method will automatically call the ``setup`` of the current module
    and recursively call ``setup`` methods of all submodules.
    """

    def setup(self, key: chex.PRNGKey = None) -> State:
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

    def init(self, key: chex.PRNGKey = None, name="_top_level") -> State:
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
            if not attr_name.startswith("_") and isinstance(attr, Module):
                if key is None:
                    subkey = None
                else:
                    key, subkey = jax.random.split(key)
                submodule_name = f"_submodule_{attr_name}"
                submodule_state = attr.init(subkey, submodule_name)
                assert isinstance(
                    submodule_state, State
                ), "setup method must return a State"
                child_states[submodule_name] = submodule_state
        self_state = self.setup(key)
        return self_state._set_child_states(child_states)
