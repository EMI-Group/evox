import chex
import jax
import types
from functools import partial


def jit(func):
    return jax.jit(func, static_argnums=(0,))


def use_state(func):
    """Decorator for easy state management.

    Parameters
    ----------
    func
        The method to be wrapped with
    """

    def wrapper(self, state, *args, **kargs):
        if self.name == "_top_level":
            return func(self, state, *args, **kargs)
        else:
            return_value = func(self, state[self.name], *args, **kargs)
            if isinstance(return_value, tuple):
                state = {**state, **{self.name: return_value[0]}}
                return state, *return_value[1:]
            else:
                return {**state, **{self.name: return_value}}

    return wrapper


def vmap_method(method):
    def wrapped(self, *args, **kargs):
        return jax.vmap(partial(method, self))(*args, **kargs)

    return wrapped


def vmap_setup(setup_method, n):
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


def use_state_class(cls, ignore=["setup", "init", "__init__"], ignore_prefix="_"):
    return _class_decorator(cls, use_state, ignore, ignore_prefix)


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
            elif key.startswith("_") or key in ignore:
                wrapped[key] = value
            elif callable(value):
                wrapped[key] = use_state(value)
        return super().__new__(cls, name, bases, wrapped)


class Module(metaclass=MetaModule):
    """Base class for all EvoXLib modules.

    This module allow easy managing of states.
    """

    def setup(self, key: chex.PRNGKey = None):
        """Setup mutable state here

        The state it self is immutable, but it act as a mutable state
        by returning new state each time.

        Parameters
        ----------
        key : PRNGKey
            A PRNGKey.

        Returns
        -------
        dict
            The state of this algorithm.
        """
        return {}

    def init(self, key: chex.PRNGKey = None, name="_top_level"):
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
        dict
            The state of this module and all submodules.
        """
        self.name = name
        state = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not attr_name.startswith("_") and isinstance(attr, Module):
                if key is None:
                    subkey = None
                else:
                    key, subkey = jax.random.split(key)
                submodule_name = f"_submodule_{attr_name}"
                state[submodule_name] = attr.init(subkey, submodule_name)
        return self.setup(key) | state
