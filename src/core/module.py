import types
import inspect
import keyword
from functools import wraps, partial
from typing import Callable, Optional, Sequence, Union, List, Dict, Tuple

import torch
from torch import nn


def jit(func: Callable,
        trace: bool = True,
        lazy: bool = True,
        example_inputs: Optional[Union[Tuple, Dict]] = None) -> Callable:
    """Just-In-Time (JIT) compile the given `func` via [`torch.jit.trace`](https://pytorch.org/docs/stable/generated/torch.jit.script.html) (`trace=True`) and [`torch.jit.script`](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) (`trace=False`).

    This function wrapper effectively deals with nested JIT and vector map (`vmap`) expressions like `jit(func1)` -> `vmap` -> `jit(func2)`,
    preventing possible errors.
    
    ## Notice:
        1. With `trace=True`, `torch.jit.trace` cannot use SAME example input arguments for function of DIFFERENT parameters,
        e.g., you cannot pass `tensor_a, tensor_a` to `torch.jit.trace`d version of `f(x: torch.Tensor, y: torch.Tensor)`.
        2. With `trace=False`, `torch.jit.script` cannot contain `vmap` expressions directly, please wrap them with `jit(..., trace=True)` or `torch.jit.trace`.
        

    Args:
        func (`Callable`): The target function to be JIT
        trace (`bool`, optional): Whether using `torch.jit.trace` or `torch.jit.script` to JIT. Defaults to True.
        lazy (`bool`, optional): Whether JIT lazily or immediately. Defaults to True.
        example_inputs (`tuple | dict`, optional): When `trace=True` and `lazy=False`, the example inputs must be provided immediately, otherwise ignored.

    Returns:
        `Callable`: The JIT version of `func`
    """
    assert trace in [True, False]
    if not trace:
        return torch.jit.script_if_tracing(func) if lazy else torch.jit.script(func)
    elif not lazy:
        assert example_inputs is not None
        if isinstance(example_inputs, tuple):
            return torch.jit.trace(func, example_inputs)
        else:
            return torch.jit.trace(func, example_kwarg_inputs=example_inputs)
    
    args_specs = inspect.getfullargspec(func)
    jit_func = None
    
    def wrapper(*args, **kwargs):
        nonlocal jit_func
        if torch.jit.is_tracing():
            jit_func = func
            return func(*args, **kwargs)
        if not jit_func:
            jit_func = torch.jit.trace(func, example_kwarg_inputs={**dict(zip(args_specs.args, args)), **kwargs})
        return jit_func(*args, **kwargs)
    
    if hasattr(func, "__self__"):
        assert args_specs.args[0] == "self", f"Expect first argument to be self, got {args_specs.args[0]}"
        return wraps(func)(partial(wrapper, func.__self__))
    else:
        return wraps(func)(wrapper)


def jit_class(cls):
    """A helper function used to JIT decorators all member methods of class `cls`.

    Args:
        cls (`type`): The original class whose member methods are to be lazy JIT.
    
    Returns:
        The wrapped class.
        
    ## Usage:
    ```
    @jit_class
    class Example(ModuleBase):
        def __init__(self, threshold = 0.5):
            super().__init__()
            self.threshold=threshold
        
        @partial(jit, trace=False)
        def s(x: torch.Tensor, threshold: float) -> torch.Tensor:
            if x.flatten()[0] > threshold:
                return torch.sin(x)
            else:
                return torch.tan(x)
        
        def h(self, q: torch.Tensor) -> torch.Tensor:
            x = Example.s(q, self.threshold)
            return x * x.shape[1]
        
        def g(self, p: torch.Tensor) -> torch.Tensor:
            x = torch.cos(p)
            return x * p.shape[0]
            
    exp = Example(0.75)
    print(exp.h(torch.rand(10, 2)))
    print(exp.g(torch.rand(10)))
    # equivalent to
    exp = torch.jit.trace_module(Example(0.75), {"h": torch.zeros(10, 2), "g": torch.zeros(10)})
    print(exp.h(torch.rand(10, 2)))
    print(exp.g(torch.rand(10)))
    ```
    """
    _original_methods = {}
    _method_argnames = {}
    _trace_method_args = {}
    _traced_module = None
    for name, method in cls.__dict__.items():
        if callable(method):
            args_specs = inspect.getfullargspec(method)
            if len(args_specs.args) == 0 or args_specs.args[0] != "self":
                continue
            _original_methods[name] = method
            _method_argnames[name] = args_specs.args[1:]

    class WrappedModuleType(type):
        def __getattr__(cls_new, key: str):
            return getattr(cls, key)

    class WrappedModule(metaclass=WrappedModuleType):
        def __init__(self, *args, **kwargs):
            self._module = cls(*args, **kwargs)

        def __getattribute__(self, name):
            nonlocal _original_methods, _trace_method_args, _traced_module
            if name == "_module":
                return super().__getattribute__(name)
            attr = object.__getattribute__(self._module, name)

            if name in _original_methods and callable(attr):
                def method_wrapper(*args, **kwargs):
                    if name not in _trace_method_args:
                        _trace_method_args[name] = {**dict(zip(_method_argnames[name], args)), **kwargs}
                        traced_module = torch.jit.trace_module(self._module, _trace_method_args, example_inputs_is_kwarg=True)
                        _traced_module = traced_module
                    return _traced_module.__getattr__(name)(*args, **kwargs)
                return method_wrapper
            else:
                return attr

    return WrappedModule


def _if_none(a, b):
    return b if a is None else a


class ModuleBase(nn.Module):
    """
    The base module for all algorithms and problems in the library.

    ## Notice
    1. This module is an object-oriented one that can contain mutable values.
    2. Functional programming model is supported via `self.state_dict()` and `self.load_state_dict(...)`.
    
    ## Usage
    1. Static methods to be JIT shall be defined as is, e.g.,
    ```
    @jit
    def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
    ```
    2. If a member function with python dynamic control flows like `if` were to be JIT, a separated static method with `jit(..., trace=False)` or `torch.jit.script_if_tracing` shall be used:
    ```
    def ExampleModule(ModuleBase):
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
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval()
    
    def load_state_dict(self, state_dict, strict=True, assign=True):
        return super().load_state_dict(state_dict, strict, assign)
    
    def define_mutable(
        self,
        name: str,
        value: Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]],
    ) -> None:
        """Define a mutable value in this module that can be accessed via `self.[name]` and modified in-place.

        Args:
            name (`str`): The mutable value's name.
            value (`torch.Tensor | Tuple[torch.Tensor, ...], List[torch.Tensor], Dict[str, torch.Tensor]`): The mutable value, can be a tuple, list, dictionary of a `torch.Tensor`.

        Raises:
            NotImplementedError: If the mutable value's type is not supported yet.
            AssertionError: If the `name` is invalid.
        """
        assert name.isdigit() or keyword.iskeyword(name), f"Name {name} is not a valid Python name."
        if isinstance(value, torch.Tensor):
            setattr(self, name, nn.Buffer(value))
        elif isinstance(value, tuple) or isinstance(value, list):
            sub_module = ModuleBase()
            for i, v in enumerate(value):
                sub_module.define_mutable(str(i), v)
            self.add_module(name, sub_module)
        elif isinstance(value, dict):
            sub_module = ModuleBase()
            for k, v in value.items():
                sub_module.define_mutable(k, v)
            self.add_module(name, sub_module)
        else:
            raise NotImplementedError(f"Mutable of type {type(value)} is not supported yet.")
    
    def __getitem__(self, key: Union[int, slice]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Get the mutable value(s) stored in this list-like module.

        Args:
            key (Union[int, slice]): The key used to index mutable value(s).

        Raises:
            IndexError: If `key` is out of range.
            TypeError: If `key` is neither a integer nor a slice.

        Returns:
            `Union[torch.Tensor, List[torch.Tensor]]`: The indexed mutable value(s).
        """
        buffers = list(self.named_buffers(recurse=False))
        if isinstance(key, slice):
            key = range(_if_none(key.start, 0), key.stop, _if_none(key.step, 1))
            return [self.__getitem__(k) for k in key]
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(buffers)
            if key < 0 or key >= len(buffers):
                raise IndexError(f"The index {key} is out of range.")
            for k, v in buffers:
                if k == str(key):
                    return v
            raise IndexError(f"The index {key} not found.")
        else:
            raise TypeError(f"Invalid argument type {type(val)}.")
    
    def __setitem__(self, value: Union[torch.Tensor, Sequence[torch.Tensor]], key: Union[slice, int]) -> None:
        targets = self.__getitem__(key)
        if isinstance(targets, list):
            value = list(value)
            assert len(value) == len(targets), f"Length of value mismatch, expected {len(targets)}, got {len(value)}"
            for t, v in zip(targets, value):
                t.set_(v)
        else:
            assert isinstance(value, torch.Tensor), f"Type of value mismatch, expected torch.Tensor, got {type(value)}"
            targets.set_(value)
