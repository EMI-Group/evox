from abc import ABC
import inspect
import types
from functools import wraps
from typing import Callable, Sequence, Union, List, Dict, Any

import torch
from torch import nn


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
        self.train(False)
    
    def load_state_dict(self, state_dict, strict=True, assign=True):
        return super().load_state_dict(state_dict, strict, assign)
    
    def add_mutable(
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
        assert name.isdigit() or str.isidentifier(name), f"Name {name} is not a valid Python name."
        if isinstance(value, torch.Tensor):
            setattr(self, name, nn.Buffer(value))
        elif isinstance(value, tuple) or isinstance(value, list):
            sub_module = ModuleBase()
            for i, v in enumerate(value):
                sub_module.add_mutable(str(i), v)
            self.add_module(name, sub_module)
        elif isinstance(value, dict):
            sub_module = ModuleBase()
            for k, v in value.items():
                assert isinstance(k, str), f"Mutable with name type {type(k)} is not supported yet."
                sub_module.add_mutable(k, v)
            self.add_module(name, sub_module)
        else:
            raise NotImplementedError(f"Mutable of type {type(value)} is not supported yet.")
    
    def __getitem__(self, key: Union[int, slice, str]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Get the mutable value(s) stored in this list-like module.

        Args:
            key (`int | slice | str`): The key used to index mutable value(s).

        Raises:
            IndexError: If `key` is out of range.
            TypeError: If `key` is of wrong type.

        Returns:
            `torch.Tensor | List[torch.Tensor]`: The indexed mutable value(s).
        """
        buffers = list(self.named_buffers(recurse=False))
        if isinstance(key, slice):
            key = range(_if_none(key.start, 0), key.stop, _if_none(key.step, 1))
            return [self.__getitem__(k) for k in key]
        elif isinstance(key, int):
            if key < 0:
                raise IndexError(f"The index {key} cannot be negative.")
            return self.get_buffer(str(key))
        elif isinstance(key, str):
            return self.get_buffer(key)
        else:
            raise TypeError(f"Invalid argument type {type(key)}.")
    
    def __setitem__(self, value: Union[torch.Tensor, Sequence[torch.Tensor]], key: Union[slice, int]) -> None:
        """Set the mutable value(s) stored in this list-like module.

        Args:
            value (`torch.Tensor | Sequence[torch.Tensor]`): The new mutable value(s).
            key (`int | slice`): The key used to index mutable value(s).
        """
        targets = self.__getitem__(key)
        if isinstance(targets, list):
            value = list(value)
            assert len(value) == len(targets), f"Length of value mismatch, expected {len(targets)}, got {len(value)}"
            for t, v in zip(targets, value):
                t.set_(v)
        else:
            assert isinstance(value, torch.Tensor), f"Type of value mismatch, expected torch.Tensor, got {type(value)}"
            targets.set_(value)


global _using_state
_using_state = False


class UseStateContext:
    
    def __init__(self, new_use_state: bool = True):
        global _using_state
        self.prev = _using_state
        self.now = new_use_state
    
    def __enter__(self):
        global _using_state
        _using_state = self.now
    
    def __exit__(self, *args):
        global _using_state
        _using_state = self.prev
    
    @staticmethod
    def is_using_state():
        global _using_state
        return _using_state


def tracing_or_using_state():
    return torch.jit.is_tracing() or UseStateContext.is_using_state()


class _WrapClassBase(ABC):
    pass


def use_state(func: Callable, is_generator: bool = False) -> Callable:
    with UseStateContext():
        # get function closure
        if is_generator:
            func = func()
        vars = inspect.getclosurevars(func)
        vars = {**vars.globals, **vars.nonlocals}
        vars = {k: v for k, v in vars.items() if isinstance(v, nn.Module)}
        # remove duplicate self
        if hasattr(inspect.unwrap(func), "__self__") and "self" in vars:
            self_v = vars["self"]
            if isinstance(self_v, _WrapClassBase):
                self_v = self_v.__inner_module__
            vars = {k: v for k, v in vars.items() if v != self_v}
            vars["self"] = self_v
        # get module states
        modules = {}
        for k, v in vars.items():
            v.state_dict(destination=modules, prefix=k + ".")
        
        
        @wraps(func)
        def wrapper(state: Dict[str, torch.Tensor], *args, **kwargs):
            with UseStateContext():
                for k, v in vars.items():
                    this_state = {".".join(key.split(".")[1:]): val for key, val in state.items() if key.split(".")[0] == k}
                    if len(this_state) > 0:
                        v.load_state_dict(this_state, assign=True)
                
                ret = func(*args, **kwargs)
                
                mutable_modules = {}
                for k, v in vars.items():
                    v.state_dict(destination=mutable_modules, prefix=k + ".")
                
                return mutable_modules, ret
        
        wrapper.init_state = modules
        return wrapper


_TRACE_WRAP_NAME = "__trace_wrapped__"
global _TORCHSCRIPT_MODIFIER
_TORCHSCRIPT_MODIFIER = "_torchscript_modifier"


def trace_impl(target: Callable):
    """A helper function used to annotate that the wrapped method shall be treated as a trace-JIT-time proxy of the given `target` method.
    
    Can ONLY be used inside a `jit_class` for a member method.
    
    ## Notice:
    The target function and the annotated function MUST have same input/output signatures (e.g. number of arguments and types); otherwise, the resulting behavior is UNDEFINED.

    Args:
        target (`Callable`): The target method invoked when not tracing JIT.

    Returns:
        The wrapping function to annotate the member method.
    """
    # deal with torchscript_modifier in case the implementation changed
    global _TORCHSCRIPT_MODIFIER
    torch.jit.export(target)
    for k in target.__dict__.keys():
        if "torch" in k and "modifier" in k:
            _TORCHSCRIPT_MODIFIER = k
            break
    
    def wrapping_fn(func: Callable) -> Callable:
        torch.jit.ignore(func)
        setattr(func, _TRACE_WRAP_NAME, target)
        return func
    
    return wrapping_fn


def jit_class(cls, trace=False):
    """A helper function used to JIT script (`torch.jit.script`) or trace (`torch.jit.trace_module`) all member methods of class `cls`.

    Args:
        cls (`type`): The original class whose member methods are to be lazy JIT.
    
    Returns:
        The wrapped class.
    
    ## Notice:
    1. In many cases, it is not necessary to wrap your custom algorithms or problems with `jit_class`, the workflow(s) will do the trick for you.
    2. With `trace=True`, all the member functions are effectively modified to return `self` additionally since side-effects cannot be traced.
    
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
    assert issubclass(cls, nn.Module), f"Expect the wrapping class to inherit `torch.nn.Module`, got {cls}"
    
    _original_methods = {}
    _trace_correspond_methods = {}
    _method_argnames = {}
    _trace_method_args = {}
    for name, method in cls.__dict__.items():
        if not callable(method):
            continue
        method_sign = inspect.signature(method)
        if len(method_sign.parameters) == 0 or "self" not in method_sign.parameters:
            continue
        if name.startswith("__") and name.endswith("__"):
            continue
        if hasattr(method, _TRACE_WRAP_NAME):
            _trace_correspond_methods[getattr(method, _TRACE_WRAP_NAME).__name__] = method
            continue
        if not trace:
            torch.jit.export(method)
        _original_methods[name] = method
        _method_argnames[name] = list(method_sign.parameters.keys())[1:]
    
    class WrappedModuleType(type(_WrapClassBase)):
        
        def __getattr__(cls_new, name):
            return getattr(cls, name)
        
        def __setattr__(cls_new, name, value):
            return setattr(cls, name, value)
    
    @wraps(cls, updated=())
    class WrappedModule(_WrapClassBase, metaclass=WrappedModuleType):
        
        def __init__(self, *args, **kwargs):
            self.__inner_module__ = cls(*args, **kwargs)
            self.__jit_module__ = None
        
        def __str__(self) -> str:
            return object.__str__(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__)
        
        def __repr__(self) -> str:
            return object.__repr__(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__)
        
        def __hash__(self) -> int:
            return object.__hash__(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__)
        
        def __format__(self, format_spec: str) -> str:
            return object.__format__(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__,
                                     format_spec)
        
        def __getitem__(self, key):
            return (self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__).__getitem__(key)
        
        def __setitem__(self, value, key):
            (self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__).__setitem__(value, key)
        
        def __setattr__(self, name, value):
            if name not in ["__inner_module__", "__jit_module__"]:
                setattr(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__, name, value)
            else:
                object.__setattr__(self, name, value)
        
        def __delattr__(self, name, value):
            if name not in ["__inner_module__", "__jit_module__"]:
                delattr(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__, name, value)
            else:
                object.__delattr__(self, name, value)
        
        def __getattribute__(self, name):
            nonlocal _original_methods, _trace_method_args
            if name in ["__inner_module__", "__jit_module__"]:
                return object.__getattribute__(self, name)
            jit_mod: torch.jit.ScriptModule = self.__jit_module__
            org_mod: nn.Module = self.__inner_module__
            if name not in _original_methods:
                if jit_mod is None:
                    return getattr(org_mod, name)
                else:
                    return getattr(jit_mod, name) if hasattr(jit_mod, name) else getattr(org_mod, name)
            
            # deal with script
            if not trace and not tracing_or_using_state():
                if jit_mod is None:
                    self.__jit_module__ = torch.jit.script(org_mod)
                return self.__jit_module__.__getattr__(name)
            
            # deal with pure trace
            func = getattr(org_mod, name)
            
            @wraps(func)
            def method_wrapper(*args, **kwargs):
                if tracing_or_using_state():
                    if not trace and name in _trace_correspond_methods:
                        if hasattr(_trace_correspond_methods[name], "__self__"):
                            bounded_trace_target_func = _trace_correspond_methods[name]
                        else:
                            bounded_trace_target_func = types.MethodType(_trace_correspond_methods[name], org_mod)
                            _trace_correspond_methods[name] = bounded_trace_target_func
                        func = bounded_trace_target_func
                    return func(*args, **kwargs)
                # else, the outer-most method is tracing
                if name not in _trace_method_args:
                    _trace_method_args[name] = {**dict(zip(_method_argnames[name], args)), **kwargs}
                    self.__jit_module__ = torch.jit.trace_module(org_mod, _trace_method_args, example_inputs_is_kwarg=True)
                return getattr(self.__jit_module__, name)(*args, **kwargs)
            
            return method_wrapper
    
    return WrappedModule


# Test
if __name__ == "__main__":
    
    @jit_class
    class Test(ModuleBase):
        
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
            return x * x.shape[1]
        
        @trace_impl(h)
        def th(self, q: torch.Tensor) -> torch.Tensor:
            x = torch.where(q.flatten()[0] > self.threshold, q + 2, q + 5)
            x += self.g(x).abs()
            x *= x.shape[1]
            self.sub_mod.buf = x.sum()
            return x
        
        def g(self, p: torch.Tensor) -> torch.Tensor:
            x = torch.cos(p)
            return x * p.shape[0]
    
    t = Test()
    print(t.h.inlined_graph)
    result = t.g(torch.randn(100, 4))
    print(result)
    t.add_mutable("mut_list", [torch.zeros(10), torch.ones(10)])
    t.add_mutable("mut_dict", {"a": torch.zeros(20), "b": torch.ones(20)})
    print(t.mut_list[0])
    print(t.mut_dict["b"])
    
    t = Test()
    fn = use_state(lambda: t.h, is_generator=True)
    trace_fn = torch.jit.trace(fn, (fn.init_state, torch.ones(10, 1)), strict=False)
    
    def loop(init_state: Dict[str, torch.Tensor], init_x: torch.Tensor, n: int = 1000):
        state = init_state
        ret = init_x
        rets: List[torch.Tensor] = []
        for _ in range(n):
            state, ret = trace_fn(state, ret)
            rets.append(state["self.sub_mod.buf"])
        return rets
    
    print(trace_fn.code)
    loop = torch.jit.trace(loop, (fn.init_state, torch.rand(10, 2)), strict=False)
    print(loop.code)
    print(loop(fn.init_state, torch.rand(10, 2)))
