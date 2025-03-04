import inspect
import types
from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn as nn

from ..core import _vmap_fix

_WRAPPING_MODULE_NAME = "__wrapping_module__"


def _if_none(a, b):
    return b if a is None else a


def _is_magic(name: str):
    return name.startswith("__") and name.endswith("__")


ParameterT = TypeVar("ParameterT", torch.Tensor, int, float, complex)


def Parameter(
    value: ParameterT,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> ParameterT:
    """Wraps a value as parameter with `requires_grad=False`.

    :param value: The parameter value.
    :param dtype: The dtype of the parameter. Defaults to None.
    :param device: The device of the parameter. Defaults to None.
    :param requires_grad: Whether the parameter requires gradient. Defaults to False.

    :return: The parameter.
    """
    return nn.Parameter(
        (
            value.to(dtype=dtype, device=device)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, dtype=dtype, device=device)
        ),
        requires_grad=requires_grad,
    )


def Mutable(value: torch.Tensor, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Wraps a value as a mutable tensor.

    :param value: The value to be wrapped.
    :param dtype: The dtype of the tensor. Defaults to None.
    :param device: The device of the tensor. Defaults to None.

    :return: The wrapped tensor.
    """
    return nn.Buffer(value.to(dtype=dtype, device=device))


def assign_load_state_dict(self: nn.Module, state_dict: Mapping[str, torch.Tensor]):
    """Copy parameters and buffers from state_dict into this module and its descendants.

    This method is used to mimic the behavior of `ModuleBase.load_state_dict` so that a regular `nn.Module` can be used with `vmap`.

    ## Usage:
    ```
    import types
    # ...
    model = ... # define your model
    model.load_state_dict = types.MethodType(assign_load_state_dict, model)
    vmap_forward = vmap(use_state(model.forward))
    jit_forward = jit(vmap_forward, trace=True, example_inputs=(vmap_forward.init_state(), ...)) # JIT trace forward pass of the model
    ```
    """
    assert not isinstance(self, ModuleBase), "Cannot call `assign_load_state_dict` on `ModuleBase`"
    sub_modules: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in state_dict.items():
        if "." in k:
            sub_key, sub_mod = ((t := k.split(".", 1))[1], t[0])
            if sub_mod not in sub_modules:
                sub_modules[sub_mod] = {}
            sub_modules[sub_mod][sub_key] = v
        else:
            setattr(self, k, v)
    if len(sub_modules) > 0:
        for k, v in sub_modules.items():
            assign_load_state_dict(getattr(self, k), v)


class ModuleBase(nn.Module):
    """
    The base module for all algorithms and problems in the library.

    ## Notice
    1. This module is an object-oriented one that can contain mutable values.
    2. Functional programming model is supported via `self.state_dict(...)` and `self.load_state_dict(...)`.
    3. The module initialization for non-static members are recommended to be written in the overwritten method of `setup` (or any other member method) rather than `__init__`.
    4. Basically, predefined submodule(s) which will be ADDED to this module and accessed later in member method(s) should be treated as "non-static members", while any other member(s) should be treated as "static members".

    ## Usage
    1. Static methods to be JIT shall be defined as is, e.g.,
    ```
    @jit
    def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
    ```
    2. If a class member function with python dynamic control flows like `if` were to be JIT, a separated static method with `jit(..., trace=False)` or `torch.jit.script_if_tracing` shall be used:
    ```
    class ExampleModule(ModuleBase):
        def setup(self, mut: torch.Tensor):
            self.add_mutable("mut", mut)
            # or
            self.mut = Mutable(mut)
            return self

        @partial(jit, trace=False)
        def static_func(x: torch.Tensor, threshold: float) -> torch.Tensor:
            if x.flatten()[0] > threshold:
                return torch.sin(x)
            else:
                return torch.tan(x)
        @jit
        def jit_func(self, p: torch.Tensor) -> torch.Tensor:
            x = ExampleModule.static_func(p, self.threshold)
            ...
    ```
    3. `ModuleBase` is usually used with `jit_class` to automatically JIT all non-magic member methods:
    ```
    @jit_class
    class ExampleModule(ModuleBase):
        # This function will be automatically JIT
        def func1(self, x: torch.Tensor) -> torch.Tensor:
            pass

        # Use `torch.jit.ignore` to disable JIT and leave this function as Python callback
        @torch.jit.ignore
        def func2(self, x: torch.Tensor) -> torch.Tensor:
            # you can implement pure Python logic here
            pass

        # JIT functions can invoke other JIT functions as well as non-JIT functions
        def func3(self, x: torch.Tensor) -> torch.Tensor:
            y = self.func1(x)
            z = self.func2(x)
            pass
    ```
    """

    def __init__(self, *args, **kwargs):
        """Initialize the ModuleBase.

        :param *args: Variable length argument list, passed to the parent class initializer.
        :param **kwargs: Arbitrary keyword arguments, passed to the parent class initializer.

        Attributes:
            __static_names__ (list): A list to store static member names.
        """

        super().__init__(*args, **kwargs)
        self.train(False)
        self.__static_names__ = []

    def eval(self):
        assert False, "`ModuleBase.eval()` shall never be invoked to prevent ambiguity."

    def setup(self, *args, **kwargs):
        """Setup the module.
        Module initialization lines should be written in the overwritten method of `setup` rather than `__init__`.

        :return: This module.

        ## Notice
        The static initialization can still be written in the `__init__` while the mutable initialization cannot.
        Therefore, multiple calls of `setup` for multiple initializations are possible.
        """
        if hasattr(self, _WRAPPING_MODULE_NAME):
            wrapper: _WrapClassBase = object.__getattribute__(self, _WRAPPING_MODULE_NAME)
            wrapper.__jit_module__ = None
        return self

    def prepare_control_flow(
        self, *target_functions: Callable, keep_vars: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[List[str], List[str]]]:
        """Prepares the control flow state of the module by collecting and merging the state and non-local variables from the specified target functions.

        This function is used alongside with `after_control_flow()` to enable your control flow operations (`utils.control_flow.*`) deal with side-effects correctly. If the control flow operations have NO side-effects, you can safely ignore this function and `after_control_flow()`.

        :param target_functions: Functions whose non-local variables are to be collected.
        :param keep_vars: See `torch.nn.Module.state_dict(..., keep_vars)`. Defaults to True.

        :return: A tuple containing the merged state dictionary, a list of state keys, and a list of non-local variable names.

        :raises AssertionError: If not all target functions are local, global, or this class member functions

        ## Warning
        The non-local variables collected here can ONLY be used as read-only ones. In-place modifications to these variables may not raise any error and silently produce incorrect results.

        ## Usage
        ```
        @jit_class
        def ExampleModule(ModuleBase):
            # define the normal `__init__` and `test` functions, etc.

            @trace_impl(test)
            def trace_test(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                self.q = self.q + 1
                local_q = self.q * 2

                def false_branch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    # nonlocal local_q ## These two lines may silently produce incorrect results
                    # local_q *= 1.5
                    return x * y * local_q # However, using read-only nonlocals is allowed

                state, keys = self.prepare_control_flow(self.true_branch, false_branch)
                if not hasattr(self, "_switch_"):
                    self._switch_ = TracingSwitch(self.true_branch, false_branch)
                state, ret = self._switch_.switch(state, (x.flatten()[0] > 0).to(dtype=torch.int), x, y)
                self.after_control_flow(state, *keys)
                return ret
        ```
        """
        state = {}
        self.state_dict(destination=state, prefix="self.", keep_vars=keep_vars)
        nonlocal_vars = {}
        for f in target_functions:
            _, _, new_vars, is_empty_vars = _get_vars(f, self, is_generator=False)
            if not is_empty_vars:
                nonlocal_vars.update(new_vars)
        if len(nonlocal_vars) == 0:
            nonlocal_vars[_EMPTY_NAME] = torch.empty(0)
        for key in nonlocal_vars:
            assert "self." not in key, "Expect all target functions are local, global, or this class member functions"
        state_keys = list(state.keys())
        nonlocal_keys = list(nonlocal_vars.keys())
        state.update(nonlocal_vars)
        return state, (state_keys, nonlocal_keys)

    def after_control_flow(
        self, state: Dict[str, torch.Tensor], state_keys: List[str], nonlocal_keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Restores the module state to the one before `prepare_control_flow` from the given `state` and returns the non-local variables collected in `prepare_control_flow`.

        This function is used alongside with `prepare_control_flow()` to enable your control flow operations (`utils.control_flow.*`) deal with side-effects correctly. If the control flow operations have NO side-effects, you can safely ignore this function and `prepare_control_flow()`.

        :param state: The state dictionary to restore the module state from.
        :param state_keys: The keys of the state dictionary that represent the module state.
        :param nonlocal_keys: The keys of the state dictionary that represent the non-local variables.

        :return: The non-local variables dictionary collected in `prepare_control_flow`.

        ## Usage
        See `prepare_control_flow()`.
        """
        locals_vars = {k: state[k] for k in nonlocal_keys if k in state}
        ori_state = {k.replace("self.", "", 1): state[k] for k in state_keys if k in state}
        self.load_state_dict(ori_state)
        return locals_vars

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], copy: bool = False, **kwargs):
        """Copy parameters and buffers from state_dict into this module and its descendants.
        Overwrites [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict).

        :param state_dict: A dict containing parameters and buffers used to update this module. See `torch.nn.Module.load_state_dict`.
        :param copy: Use the original `torch.nn.Module.load_state_dict` to copy the `state_dict` to current state (`copy=True`) or use this implementation that assigns the values of this module to the ones in the `state_dict` (`copy=False`). Defaults to False.
        :param **kwargs: The original arguments of `torch.nn.Module.load_state_dict`. Ignored if `copy=False`.

        :return: If `copy=True`, returns the return of `torch.nn.Module.load_state_dict`; otherwise, no return.
        """
        if copy:
            assert not any(map(_vmap_fix.is_batched_tensor, state_dict.values())), "`copy=True` is not compatible with `vmap`"
            return super().load_state_dict(state_dict, **kwargs)
        # else
        sub_modules: Dict[str, Dict[str, torch.Tensor]] = {}
        for k, v in state_dict.items():
            if "." in k:
                sub_key, sub_mod = ((t := k.split(".", 1))[1], t[0])
                if sub_mod not in sub_modules:
                    sub_modules[sub_mod] = {}
                sub_modules[sub_mod][sub_key] = v
            else:
                if isinstance(self.__getattr_inner__(k), nn.Parameter) and not isinstance(v, nn.Parameter):
                    v = nn.Parameter(v, requires_grad=v.requires_grad)
                self.__setattr_inner__(k, v)
        if len(sub_modules) > 0:
            for k, v in sub_modules.items():
                getattr(self, k).load_state_dict(v)

    def add_mutable(
        self,
        name: str,
        value: Union[
            torch.Tensor | nn.Module,
            Sequence[torch.Tensor | nn.Module],
            Dict[str, torch.Tensor | nn.Module],
        ],
    ) -> None:
        """Define a mutable value in this module that can be accessed via `self.[name]` and modified in-place.

        :param name: The mutable value's name.
        :param value: The mutable value, can be a tuple, list, dictionary of a `torch.Tensor`.

        :raises NotImplementedError: If the mutable value's type is not supported yet.
        :raises AssertionError: If the `name` is invalid.
        """
        assert name.isdigit() or str.isidentifier(name), f"Name {name} is not a valid Python name."
        if isinstance(value, torch.Tensor):
            setattr(self, name, nn.Buffer(value))
        elif isinstance(value, nn.Module):
            super().__setattr__(name, value)
        elif isinstance(value, tuple) or isinstance(value, list):
            all_tensors = all(map(lambda x: isinstance(x, torch.Tensor), value))
            all_modules = all(map(lambda x: isinstance(x, nn.Module), value))
            assert all_tensors or all_modules, (
                "Expect a tuple or list of `torch.Tensor` or `nn.Module`, got a mixture of both or none of them."
            )
            if all_modules:
                sub_module = nn.ModuleList(value)
                self.add_module(name, sub_module)
            else:
                sub_module = ModuleBase()
                for i, v in enumerate(value):
                    sub_module.add_mutable(str(i), v)
                self.add_module(name, sub_module)
        elif isinstance(value, dict):
            all_tensors = all(map(lambda x: isinstance(x, torch.Tensor), value.values()))
            all_modules = all(map(lambda x: isinstance(x, nn.Module), value.values()))
            assert all_tensors or all_modules, (
                "Expect a dict of `torch.Tensor` or `nn.Module`, got a mixture of both or none of them."
            )
            if all_modules:
                sub_module = nn.ModuleDict(value)
                self.add_module(name, sub_module)
            else:
                sub_module = ModuleBase()
                for k, v in value.items():
                    sub_module.add_mutable(k, v)
                self.add_module(name, sub_module)
        else:
            raise NotImplementedError(f"Mutable of type {type(value)} is not supported yet.")

    def to(self, *args, **kwargs) -> "ModuleBase":
        super().to(*args, **kwargs)
        for k in self.__static_names__:
            val = object.__getattribute__(self, k)
            if isinstance(val, torch.Tensor):
                val = val.to(**kwargs)
                self.__setattr_inner__(k, val)
        return self

    def __getattribute__(self, name):
        if not tracing_or_using_state() or name == _WRAPPING_MODULE_NAME or _is_magic(name):
            return super(nn.Module, self).__getattribute__(name)
        self_dict = super(nn.Module, self).__getattribute__("__dict__")
        if _WRAPPING_MODULE_NAME not in self_dict:
            return super(nn.Module, self).__getattribute__(name)
        try:
            return self_dict.get(_WRAPPING_MODULE_NAME).__getattr__(name)
        except Exception:
            return super(nn.Module, self).__getattribute__(name)

    def __getattr_inner__(self, name):
        try:
            value = super(nn.Module, self).__getattribute__(name)
        except Exception:
            value = super(ModuleBase, self).__getattr__(name)
        return value

    def __delattr__(self, name):
        if name == _WRAPPING_MODULE_NAME:
            self.__delattr_inner__(name)
        elif not hasattr(self, _WRAPPING_MODULE_NAME):
            self.__delattr_inner__(name)
        else:
            object.__getattribute__(self, _WRAPPING_MODULE_NAME).__delattr__(name)

    def __delattr_inner__(self, name):
        try:
            object.__delattr__(self, name)
        except Exception:
            super(ModuleBase, self).__delattr__(name)

    def __setattr__(self, name, value):
        if type(value) is nn.Module:  # an empty nn.Module, change to ModuleBase
            super().__setattr__(name, ModuleBase())
        elif hasattr(self, _WRAPPING_MODULE_NAME):
            wrapper = object.__getattribute__(self, _WRAPPING_MODULE_NAME)
            wrapper.__setattr__(name, value)
        else:
            old_names = set(self.__dict__.keys())
            super().__setattr__(name, value)
            new_names = set(self.__dict__.keys())
            new_names.difference_update(old_names)
            new_names = list(new_names)
            if len(new_names) > 0 and not _is_magic(new_names[0]):
                self.__static_names__.append(new_names[0])

    def __setattr_inner__(self, name, value):
        super().__setattr__(name, value)

    def __getitem__(self, key: Union[int, slice, str]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Get the mutable value(s) stored in this list-like module.

        :param key: The key used to index mutable value(s).

        :raises IndexError: If `key` is out of range.
        :raises TypeError: If `key` is of wrong type.

        :return: The indexed mutable value(s).
        """
        if isinstance(key, slice):
            key = range(_if_none(key.start, 0), key.stop, _if_none(key.step, 1))
            return [self.__getitem__(k) for k in key]
        elif isinstance(key, int):
            if key < 0:
                raise IndexError(f"The index {key} cannot be negative.")
            key = str(key)
        if isinstance(key, str) and key in self._buffers:
            return self.get_buffer(key)
        else:
            raise TypeError(f"Invalid argument type {type(key)}.")

    def __setitem__(
        self,
        value: Union[torch.Tensor, List[torch.Tensor]],
        key: Union[slice, int],
    ) -> None:
        """Set the mutable value(s) stored in this list-like module.

        :param value: The new mutable value(s).
        :param key: The key used to index mutable value(s).
        """
        targets = self.__getitem__(key)
        if isinstance(targets, list):
            value = list(value)
            assert len(value) == len(targets), "Length of value mismatch, expected {len(targets)}, got {len(value)}"
            for t, v in zip(targets, value):
                t.set_(v)
        else:
            assert isinstance(value, torch.Tensor), f"Type of value mismatch, expected torch.Tensor, got {type(value)}"
            targets.set_(value)

    def iter(self) -> Tuple[torch.Tensor]:
        if len(self._buffers) > 0:
            return tuple(self.buffers(recurse=False))
        else:
            return tuple(self.modules())

    def __sync_with__(self, jit_module: torch.jit.ScriptModule | None):
        if jit_module is None:
            return
        self.load_state_dict(jit_module.state_dict(keep_vars=True))
        for k in self.__static_names__:
            try:
                self.__setattr_inner__(k, jit_module.__getattr__(k))
            except Exception:
                self.__static_names__.remove(k)
        for sub_name, sub_mod in self._modules.items():
            if len(sub_name) > 0 and isinstance(sub_mod, ModuleBase):
                sub_mod.__sync_with__(jit_module.__getattr__(sub_name))


_using_state: ContextVar[bool] = ContextVar("using_state", default=False)
_trace_caching_state: ContextVar[bool] = ContextVar("trace_caching_state", default=False)


@contextmanager
def use_state_context(new_use_state: bool = True):
    """
    A context manager to set the value of `using_state` temporarily.

    When entering the context, the value of `using_state` is set to `new_use_state` and a token is obtained.
    When exiting the context, the value of `using_state` is reset to its previous value.

    :param new_use_state: The new value of `using_state`. Defaults to True.

    ## Examples:
    ```
    >>> with use_state_context(True):
    ...     assert is_using_state()
    >>> assert not is_using_state()
    ```
    """
    # Set the new state and obtain a token
    token: Token = _using_state.set(new_use_state)
    try:
        yield token
    finally:
        # Reset the state to its previous value
        _using_state.reset(token)


@contextmanager
def trace_caching_state_context(new_trace_caching_state: bool = True):
    """
    A context manager to set the value of `trace_caching_state` temporarily.

    When entering the context, the value of `trace_caching_state` is set to `new_trace_caching_state` and a token is obtained.
    When exiting the context, the value of `trace_caching_state` is reset to its previous value.

    :param new_trace_caching_state: The new value of `trace_caching_state`. Defaults to True.

    ## Examples:
    ```
    >>> with trace_caching_state_context(True):
    ...     assert is_trace_caching_state()
    >>> assert not is_trace_caching_state()
    ```
    """
    token: Token = _trace_caching_state.set(new_trace_caching_state)
    try:
        yield token
    finally:
        _trace_caching_state.reset(token)


def is_using_state() -> bool:
    """
    Get the current state of the `using_state`.

    :return: The current state of the `using_state`.
    """
    return _using_state.get()


def is_trace_caching_state() -> bool:
    """
    Get the current state of the `trace_caching_state`.

    :return: The current state of the `trace_caching_state`.
    """
    return _trace_caching_state.get()


def tracing_or_using_state():
    """
    Check if we are currently JIT tracing (inside a `torch.jit.trace`), in a `use_state_context`, or in a `trace_caching_state`.

    :return: True if either condition is true, False otherwise.
    """
    return torch.jit.is_tracing() or is_using_state() or is_trace_caching_state()


_SUBMODULE_PREFIX = "__submodule_"


class _WrapClassBase:
    def __init__(self, inner: ModuleBase):
        assert isinstance(inner, ModuleBase), f"Inner module must be a `ModuleBase`, got {type(inner)}"
        object.__setattr__(inner, _WRAPPING_MODULE_NAME, self)
        self.__inner_module__ = inner
        self.__jit_module__: torch.jit.ScriptModule | None = None

    def __str__(self) -> str:
        return object.__str__(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__)

    def __repr__(self) -> str:
        return object.__repr__(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__)

    def __hash__(self) -> int:
        return object.__hash__(self.__inner_module__)

    def __format__(self, format_spec: str) -> str:
        return object.__format__(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__, format_spec)

    def __getitem__(self, key):
        return (self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__).__getitem__(key)

    def __setitem__(self, value, key):
        (self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__).__setitem__(value, key)

    def __setattr__(self, name, value):
        if name in ["__inner_module__", "__jit_module__"]:
            return object.__setattr__(self, name, value)
        # special treatment for compatibility with `vmap`
        value = _vmap_fix.align_vmap_tensor(value, getattr(self.__inner_module__, name, None))
        # else
        if not isinstance(value, _WrapClassBase):
            self.__inner_module__.__setattr_inner__(name, value)
            if self.__jit_module__ is not None:
                self.__jit_module__.__setattr__(name, value)
            return
        # special treatment for compatibility with `trace_impl` of sub modules
        inner_sub_mod = value.__inner_module__
        self.__inner_module__.__setattr_inner__(name, inner_sub_mod)
        object.__setattr__(self, _SUBMODULE_PREFIX + name, value)

    def __delattr__(self, name):
        if name not in ["__inner_module__", "__jit_module__"]:
            self.__inner_module__.__delattr_inner__(name)
            if name in self.__dict__:
                object.__delattr__(self, name)
            if self.__jit_module__ is not None:
                if name in self.__jit_module__._modules:
                    del self.__jit_module__._modules._python_modules[name]
                # elif name in self.__jit_module__._parameters:
                #     del self.__jit_module__._parameters._python_parameters[name]
                # elif name in self.__jit_module__._buffers:
                #     del self.__jit_module__._buffers._python_buffers[name]
                else:
                    pass  # cannot delete JIT module parameters, buffers and attributes
        else:
            object.__delattr__(self, name)

    def __sync__(self):
        self.__inner_module__.__sync_with__(self.__jit_module__)


_USE_STATE_NAME = "__use_state__"
_STATE_ARG_NAME = "state"


class UseStateFunc(Protocol):
    is_empty_state: bool

    def init_state(self, clone: bool = True) -> Dict[str, torch.Tensor]:
        """Get the cloned state of the closures of the function when it is wrapped by `use_state`.

        :param clone: Whether to clone the original state or not. Defaults to True.

        :return: The cloned state of the closures.
        """
        pass

    def set_state(self, state: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Set the closures of the function to the given state.

        :param state: The new state to set to. If `state=None`, the new state would be the original state when the function is wrapped by `use_state`. Defaults to None.
        """
        pass

    def __call__(
        self, state: Dict[str, torch.Tensor], *args, **kwargs
    ) -> Dict[str, torch.Tensor] | Tuple[Dict[str, torch.Tensor], Any]:
        pass


_EMPTY_NAME = "___empty___"


def _get_vars(func: Callable, *exclude, is_generator: bool = True):
    with use_state_context():
        if is_generator:
            func = func()
        # get function closure
        func_args = inspect.signature(func).parameters.keys()
        assert _STATE_ARG_NAME not in func_args, f"Use-state functions cannot have argument of name `{_STATE_ARG_NAME}`"
        vars = inspect.getclosurevars(func)
        vars = {**vars.globals, **vars.nonlocals}
        vars = {
            k: v
            for k, v in vars.items()
            if isinstance(v, nn.Module) or isinstance(v, _WrapClassBase) or isinstance(v, torch.Tensor)
        }
        # remove duplicate self
        if "self" in vars:  # and hasattr(inspect.unwrap(func), "__self__"):
            self_v: Tuple[nn.Module, ...] = vars["self"]
            if isinstance(self_v, _WrapClassBase):
                if self_v.__jit_module__ is None:
                    self_v = (self_v.__inner_module__,)
                else:
                    self_v = (self_v.__inner_module__, self_v.__jit_module__)
            else:
                self_v = (self_v,)
            if len(self_v) > 1:  # sync with JIT module first
                self_v[0].load_state_dict(self_v[1].state_dict(keep_vars=True))
            vars = {k: v for k, v in vars.items() if v not in self_v}
            vars["self"] = self_v[0]
        elif hasattr(func, "__self__") and isinstance(func.__self__, nn.Module):
            vars["self"] = func.__self__
        # exclude
        vars = {k: v for k, v in vars.items() if v not in exclude}
        # get module states
        modules_vars: Dict[str, torch.Tensor] = {}
        for k, v in vars.items():
            if isinstance(v, torch.Tensor):
                modules_vars[k] = v
            else:
                v.state_dict(destination=modules_vars, prefix=k + ".", keep_vars=True)
        # special case for empty state
        is_empty_state = len(modules_vars) == 0
        if is_empty_state:
            modules_vars = {_EMPTY_NAME: torch.empty(0)}
        return func, vars, modules_vars, is_empty_state


def use_state(func: Callable[[], Callable] | Callable, is_generator: bool = True) -> UseStateFunc:
    """Transform the given stateful function (which in-place alters `nn.Module`s) to a pure-functional version that receives an additional `state` parameter (of type `Dict[str, torch.Tensor]`) and returns the altered state additionally.

    :param func: The stateful function to be transformed or its generator function.
    :param is_generator: Whether `func` is a function or a function generator (e.g. a lambda that returns the stateful function). Defaults to `True`.

    :return: The transformed pure-functional version of `func`. It contains a `init_state() -> state` attribute that returns the copy of the current state that `func` uses and can be used as example inputs of the additional `state` parameter. It also contains a `set_state(state)` attribute to set the global state to the given one (of course not JIT-compatible).

    ## Notice
    Since PyTorch cannot JIT or vectorized-map a function with empty dictionary, list, or tuple as its input, this function transforms the given function to a function WITHOUT the additional `state` parameter (of type `Dict[str, torch.Tensor]`) and does NOT return the altered state additionally.

    ## Usage:
    ```
    @jit_class
    class Example(ModuleBase):

        def __init__(self, threshold=0.5):
            super().__init__()
            self.threshold = threshold
            self.sub_mod = nn.Module()
            self.sub_mod.buf = nn.Buffer(torch.zeros(()))

        def h(self, q: torch.Tensor) -> torch.Tensor:
            if q.flatten()[0] > self.threshold:
                x = torch.sin(q)
            else:
                x = torch.tan(q)
            x += self.g(x).abs()
            x *= x.shape[1]
            self.sub_mod.buf = x.sum()
            return x

        @trace_impl(h)
        def th(self, q: torch.Tensor) -> torch.Tensor:
            x += self.g(x).abs()
            x *= x.shape[1]
            self.sub_mod.buf = x.sum()
            return x

        def g(self, p: torch.Tensor) -> torch.Tensor:
            x = torch.cos(p)
            return x * p.shape[0]

    t = Test()
    fn = use_state(lambda: t.h, is_generator=True)
    jit_fn = jit(fn, trace=True, lazy=True)
    results = jit_fn(fn.init_state(), torch.rand(10, 1)
    # print results, may be
    print(results) # ({"self.sub_mod.buf": torch.Tensor(5.6)}, torch.Tensor([[0.56], ...]))
    # IN-PLACE update all relevant variables using the given state, which is the variable `t` here.
    fn.set_state(results[0])
    ```
    """
    func, vars, modules_vars, is_empty_state = _get_vars(func, is_generator=is_generator)

    @wraps(func)
    def state_wrapper(state: Dict[str, torch.Tensor], *args, **kwargs):
        with use_state_context():
            # special case for empty state
            if is_empty_state and (
                not isinstance(state, dict)
                or ((tk := tuple(state.keys())) != (_EMPTY_NAME,) and len(set(tk).difference((_EMPTY_NAME,))) > 0)
            ):
                ret = func(state, *args, **kwargs)
                return ret
            # apply new state dict
            _set_state(state)
            # get actual output
            ret = func(*args, **kwargs)
            # get changed state dict
            mutable_modules = {}
            for k, v in vars.items():
                if isinstance(v, torch.Tensor):
                    mutable_modules[k] = v
                else:
                    v.state_dict(destination=mutable_modules, prefix=k + ".", keep_vars=True)
            if len(mutable_modules) == 0:
                mutable_modules = {_EMPTY_NAME: torch.empty(0)}
            # return
            if ret is None:
                return mutable_modules
            else:
                return mutable_modules, ret

    def _set_state(state: Optional[Dict[str, torch.Tensor]] = None):
        if state is None:
            state = modules_vars

        def wrap_param_fn(key, val):
            if isinstance(modules_vars[key], nn.Parameter):
                return nn.Parameter(val, requires_grad=modules_vars[key].requires_grad)
            else:
                return val

        for k, v in vars.items():
            if isinstance(v, torch.Tensor):
                # torch.utils.swap_tensors(v, state[k])  # cannot swap or set nonlocal variables
                # v.copy_(state[k])
                continue
            this_state = {t[1]: wrap_param_fn(key, val) for key, val in state.items() if (t := key.split(".", 1))[0] == k}
            if len(this_state) > 0:
                v.load_state_dict(this_state)

    def _init_state(clone: bool = True):
        if not clone:
            return modules_vars
        state = {}
        for k, v in modules_vars.items():
            if isinstance(v, nn.Parameter):
                state[k] = nn.Parameter(v.clone(), requires_grad=v.requires_grad)
            else:
                state[k] = v.clone()
        return state

    state_wrapper.init_state = _init_state
    state_wrapper.set_state = _set_state
    state_wrapper.is_empty_state = is_empty_state
    setattr(state_wrapper, _USE_STATE_NAME, True)
    _vmap_fix._set_func_id(state_wrapper, func)
    return state_wrapper


_TORCHSCRIPT_MODIFIER = "_torchscript_modifier"

_TRACE_WRAP_NAME = "__trace_wrapped__"


T = TypeVar("T", bound=Callable)


def trace_impl(target: Callable):
    """A helper function used to annotate that the wrapped method shall be treated as a trace-JIT-time proxy of the given `target` method.

    Can ONLY be used inside a `jit_class` for a member method.

    :param target: The target method invoked when not tracing JIT.

    :return: The wrapping function to annotate the member method.

    ## Notice
    1. The target function and the annotated function MUST have same input/output signatures (e.g. number of arguments and types); otherwise, the resulting behavior is UNDEFINED.
    2. If the annotated function are to be `vmap`, it cannot contain any in-place operations to `self` since such operations are not well-defined and cannot be compiled.

    ## Usage:
    See `use_state`.
    """
    # _torch_script_modifier(target)

    def wrapping_fn(func: T) -> T:
        torch.jit.ignore(func)
        setattr(func, _TRACE_WRAP_NAME, target)
        return func

    return wrapping_fn


_VMAP_WRAP_NAME = "__vmap_wrapped__"


def vmap_impl(target: Callable):
    """A helper function used to annotate that the wrapped method shall be treated as a vmap-JIT-time proxy of the given `target` method.

    Can ONLY be used inside a `jit_class` for a member method.

    :param target: The target method invoked when not tracing JIT.

    :return: The wrapping function to annotate the member method.

    ## Notice
    1. The target function and the annotated function MUST have same input/output signatures (e.g. number of arguments and types); otherwise, the resulting behavior is UNDEFINED.
    2. If the annotated function are to be `vmap`, it cannot contain any in-place operations to `self` since such operations are not well-defined and cannot be compiled.

    ## Usage:
    See `use_state`.
    """
    # _torch_script_modifier(target)

    def wrapping_fn(func: T) -> T:
        torch.jit.ignore(func)
        setattr(func, _VMAP_WRAP_NAME, target)
        return func

    return wrapping_fn


ClassT = TypeVar("ClassT", bound=type)

_BASE_NAME = "base"


def jit_class(cls: ClassT, trace: bool = False) -> ClassT:
    """A helper function used to JIT script (`torch.jit.script`) or trace (`torch.jit.trace_module`) all member methods of class `cls`.

    :param cls: The original class whose member methods are to be lazy JIT.
    :param trace: Whether to trace the module or to script the module. Default to False.

    Returns:
        The wrapped class.

    ## Notice
    1. In many cases, it is not necessary to wrap your custom algorithms or problems with `jit_class`, the workflow(s) will do the trick for you.
    2. With `trace=True`, all the member functions are effectively modified to return `self` additionally since side-effects cannot be traced. If you want to preserve the side effects, please set `trace=False` and use the `use_state` function to wrap the member method to generate pure-functional
    3. Similarly, all module-wide operations like `self.to(...)` can only returns the unwrapped module, which may not be desired. Since most of them are in-place operations, a simple `module.to(...)` can be used instead of `module = module.to(...)`.

    ## Usage:
    ```
    @jit_class
    class Example(ModuleBase):
        # magic methods are ignored in JIT
        def __init__(self, threshold = 0.5):
            super().__init__()
            self.threshold = threshold

        # `torch.jit.export` is automatically added to this member method
        def h(self, q: torch.Tensor) -> torch.Tensor:
            if q.flatten()[0] > self.threshold:
                x = torch.sin(q)
            else:
                x = torch.tan(q)
            return x * x.shape[1]

    exp = Example(0.75)
    print(exp.h(torch.rand(10, 2)))
    # equivalent to
    exp = torch.jit.trace_module(Example(0.75))
    print(exp.h(torch.rand(10, 2)))
    ```
    """
    assert issubclass(cls, ModuleBase), f"Expect the wrapping class to inherit `ModuleBase`, got {cls}"

    _original_methods = {}
    _trace_correspond_methods = {}
    _vmap_correspond_methods = {}
    _method_argnames = {}
    _trace_method_args = {}
    for name, method in cls.__dict__.items():
        if not callable(method):
            continue
        method_sign = inspect.signature(method)
        if len(method_sign.parameters) == 0 or "self" not in method_sign.parameters:
            continue
        if name == "setup":
            continue
        if hasattr(method, _TRACE_WRAP_NAME):
            original_method = getattr(method, _TRACE_WRAP_NAME)
            _trace_correspond_methods[original_method.__name__] = {_BASE_NAME: method}
            _original_methods[original_method.__name__] = original_method
            continue
        if hasattr(method, _VMAP_WRAP_NAME):
            original_method = getattr(method, _VMAP_WRAP_NAME)
            _vmap_correspond_methods[original_method.__name__] = {_BASE_NAME: method}
            _original_methods[original_method.__name__] = original_method
            continue
        if hasattr(method, _TORCHSCRIPT_MODIFIER):
            if "export" not in getattr(method, _TORCHSCRIPT_MODIFIER):
                continue
        elif _is_magic(name):
            continue
        if not trace:
            torch.jit.export(method)
        _original_methods[name] = method
        _method_argnames[name] = list(method_sign.parameters.keys())[1:]

    class WrappedModuleType(type(_WrapClassBase)):
        def __getattr__(cls_new, name):
            return getattr(cls, name)

        def __setattr__(cls_new, name, value):
            from functools import WRAPPER_ASSIGNMENTS

            setattr(cls, name, value)
            if name in WRAPPER_ASSIGNMENTS:
                type.__setattr__(cls_new, name, value)

        def __dir__(cls_new):
            return dir(cls)

    @wraps(cls, updated=())
    class WrappedModule(_WrapClassBase, metaclass=WrappedModuleType):
        def __init__(self, *args, **kwargs):
            inner = cls(*args, **kwargs)
            super().__init__(inner)

        def __getattr__(self, name: str):
            if name in ["__inner_module__", "__jit_module__"]:
                return object.__getattribute__(self, name)

            nonlocal _original_methods, _trace_method_args
            jit_mod: torch.jit.ScriptModule = self.__jit_module__
            base_mod = self.__inner_module__

            # special treatment for compatibility with `trace_impl` of sub modules
            if tracing_or_using_state() and (_SUBMODULE_PREFIX + name) in self.__dict__:
                attr = object.__getattribute__(self, _SUBMODULE_PREFIX + name)
                if isinstance(attr, _WrapClassBase):
                    return attr

            # basic case, get from jit module or original module
            if name not in _method_argnames and (name not in _original_methods or not tracing_or_using_state()):
                if jit_mod is None or tracing_or_using_state():
                    return base_mod.__getattr_inner__(name)
                else:
                    return getattr(jit_mod, name) if hasattr(jit_mod, name) else base_mod.__getattr_inner__(name)

            # is a member method, deal with script
            if not trace and not tracing_or_using_state():
                if jit_mod is None:
                    self.__jit_module__ = torch.jit.script(base_mod)
                    object.__getattribute__(self, "__sync__")()
                return getattr(self.__jit_module__, name)

            # is a member method, deal with trace
            func = base_mod.__getattr_inner__(name)

            @wraps(func)
            def jit_member_wrapper(*args, **kwargs):
                nonlocal func
                # special treatment for compatibility with `vmap_impl`
                if _vmap_fix.current_level() is not None:
                    if not trace and name in _vmap_correspond_methods:
                        if self in _vmap_correspond_methods[name]:
                            bounded_target_func = _vmap_correspond_methods[name][self]
                        else:
                            bounded_target_func = types.MethodType(_vmap_correspond_methods[name][_BASE_NAME], self)
                            _vmap_correspond_methods[name][self] = bounded_target_func
                        func = bounded_target_func
                        return func(*args, **kwargs)
                # special treatment for compatibility with `trace_impl`
                if tracing_or_using_state():
                    if not trace and name in _trace_correspond_methods:
                        if self in _trace_correspond_methods[name]:
                            bounded_target_func = _trace_correspond_methods[name][self]
                        else:
                            bounded_target_func = types.MethodType(_trace_correspond_methods[name][_BASE_NAME], self)
                            _trace_correspond_methods[name][self] = bounded_target_func
                        func = bounded_target_func
                    return func(*args, **kwargs)
                # else, the outer-most method is tracing
                if name not in _trace_method_args:
                    _trace_method_args[name] = {**dict(zip(_method_argnames[name], args)), **kwargs}
                    self.__jit_module__ = torch.jit.trace_module(base_mod, _trace_method_args, example_inputs_is_kwarg=True)
                return getattr(self.__jit_module__, name)(*args, **kwargs)

            _vmap_fix._set_func_id(jit_member_wrapper, func)
            return jit_member_wrapper

    return WrappedModule
