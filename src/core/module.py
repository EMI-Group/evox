from abc import ABC
import inspect
import types
from functools import wraps
from typing import Mapping, Protocol, Callable, Sequence, Tuple, Union, List, Dict, Any

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
    
    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], copy: bool = False, **kwargs):
        """Copy parameters and buffers from state_dict into this module and its descendants.
        Overwrites [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict). 

        Args:
            state_dict (`Mapping[str, torch.Tensor]`): _description_
            copy (`bool`, optional): Use the original `torch.nn.Module.load_state_dict` to copy the `state_dict` to current state (`copy=True`) or use this implementation that assigns the values of this module to the ones in the `state_dict` (`copy=False`). Defaults to False.
            kwargs: The original arguments of `torch.nn.Module.load_state_dict`. Ignored 

        Returns:
            `NamedTuple | None`: If `copy=True`, returns the return of `torch.nn.Module.load_state_dict`; otherwise, no return.
        """
        if copy:
            assert any(map(torch._C._functorch.is_batchedtensor, state_dict.values())) == False, \
                    "`copy=True` is not compatible with `vmap`"
            return super().load_state_dict(state_dict, **kwargs)
        # else
        sub_modules: Dict[str, Dict[str, torch.Tensor]] = {}
        for k, v in state_dict.items():
            if '.' in k:
                sub_key = k[k.find('.') + 1:]
                sub_mod = k[:k.find('.')]
                if sub_mod not in sub_modules:
                    sub_modules[sub_mod] = {}
                sub_modules[sub_mod][sub_key] = v
            else:
                setattr(self, k, v)
        if len(sub_modules) > 0:
            for k, v in sub_modules.items():
                getattr(self, k).load_state_dict(v, copy=False)
    
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
    
    def __setattr__(self, name, value):
        if type(value) != nn.Module:
            return super().__setattr__(name, value)
        else:  # an empty nn.Module, change to ModuleBase
            return super().__setattr__(name, ModuleBase())
    
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
    
    def __init__(self, inner: nn.Module):
        self.__inner_module__ = inner
        self.__jit_module__ = None
    
    def __str__(self) -> str:
        return object.__str__(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__)
    
    def __repr__(self) -> str:
        return object.__repr__(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__)
    
    def __hash__(self) -> int:
        return object.__hash__(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__)
    
    def __format__(self, format_spec: str) -> str:
        return object.__format__(self.__inner_module__ if self.__jit_module__ is None else self.__jit_module__, format_spec)
    
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


_USE_STATE_NAME = "__use_state__"
_STATE_ARG_NAME = "state"


class UseStateFunc(Protocol):
    
    def init_state(self) -> Dict[str, torch.Tensor]:
        pass
    
    def set_state(self, state: Dict[str, torch.Tensor]) -> None:
        pass
    
    def __call__(self, state: Dict[str, torch.Tensor], *args,
                 **kwargs) -> Dict[str, torch.Tensor] | Tuple[Dict[str, torch.Tensor], Any]:
        pass


def use_state(func: Callable, is_generator: bool = False) -> UseStateFunc:
    """Transform the given stateful function (which in-place alters `nn.Module`s) to a pure-functional version that receives an additional `state` parameter (of type `Dict[str, torch.Tensor]`) and returns the altered state additionally.

    Args:
        func (`Callable`): The stateful function to be transformed.
        is_generator (`bool`, optional): Whether `func` is a function or a function generator (e.g. a lambda that returns the stateful function). Defaults to `False`.

    Returns:
        `Callable`: The transformed pure-functional version of `func`. It contains a `init_state() -> state` attribute that returns the copy of the current state that `func` uses and can be used as example inputs of the additional `state` parameter. It also contains a `set_state(state)` attribute to set the global state to the given one (of course not JIT-compatible).
        
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
    with UseStateContext():
        # get function closure
        if is_generator:
            func = func()
        func_args = inspect.signature(func).parameters.keys()
        assert _STATE_ARG_NAME not in func_args, f"Use-state functions cannot have argument of name `{_STATE_ARG_NAME}`"
        vars = inspect.getclosurevars(func)
        vars = {**vars.globals, **vars.nonlocals}
        vars = {k: v for k, v in vars.items() if isinstance(v, nn.Module)}
        # remove duplicate self
        if hasattr(inspect.unwrap(func), "__self__") and "self" in vars:
            self_v: Tuple[nn.Module, ...] = vars["self"]
            if isinstance(self_v, _WrapClassBase):
                if self_v.__jit_module__ is None:
                    self_v = (self_v.__inner_module__, )
                else:
                    self_v = (self_v.__inner_module__, self_v.__jit_module__)
            else:
                self_v = (self_v, )
            if len(self_v) > 1:
                self_v[0].load_state_dict(self_v[1].state_dict(keep_vars=True))
            vars = {k: v for k, v in vars.items() if v not in self_v}
            vars["self"] = self_v[0]
        # get module states
        modules: Dict[str, torch.Tensor] = {}
        for k, v in vars.items():
            v.state_dict(destination=modules, prefix=k + ".", keep_vars=True)
        modules = {k: v.clone() for k, v in modules.items()}
        
        @wraps(func)
        def wrapper(state: Dict[str, torch.Tensor], *args, **kwargs):
            with UseStateContext():
                # apply new state dict
                _set_state(state)
                # get actual output
                ret = func(*args, **kwargs)
                # get changed state dict
                mutable_modules = {}
                for k, v in vars.items():
                    v.state_dict(destination=mutable_modules, prefix=k + ".", keep_vars=True)
                # return
                if ret is None:
                    return mutable_modules
                else:
                    return mutable_modules, ret
        
        def _set_state(state: Dict[str, torch.Tensor]):
            for k, v in vars.items():
                this_state = {".".join(key.split(".")[1:]): val for key, val in state.items() if key.split(".")[0] == k}
                if len(this_state) > 0:
                    v.load_state_dict(this_state)
        
        wrapper.init_state = lambda: {k: v.clone() for k, v in modules.items()}
        wrapper.set_state = _set_state
        setattr(wrapper, _USE_STATE_NAME, True)
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


def jit_class(cls, trace: bool = False):
    """A helper function used to JIT script (`torch.jit.script`) or trace (`torch.jit.trace_module`) all member methods of class `cls`.

    Args:
        cls (`type`): The original class whose member methods are to be lazy JIT.
        trace (`bool`, optional): Whether to trace the module or to script the module. Default to False.
    
    Returns:
        The wrapped class.
    
    ## Notice:
    1. In many cases, it is not necessary to wrap your custom algorithms or problems with `jit_class`, the workflow(s) will do the trick for you.
    2. With `trace=True`, all the member functions are effectively modified to return `self` additionally since side-effects cannot be traced. If you want to preserve the side effects, please set `trace=False` and use the `use_state` function to wrap the member method to generate pure-functional 
    3. While direct setting of mutable values like `self.mutable_1 = new_mutable_1` can be treated by `jit_class` via explicit synchronizing the JIT module to the original module; yet, it may still cause performance and memory issues since tensors are created and destroyed frequently when calling functions like `use_state`. Therefore, we recommend using in-place operations like `self.mutable_1.copy_(new_mutable_1)`.
    
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
            inner = cls(*args, **kwargs)
            super().__init__(inner)
        
        def __getattribute__(self, name):
            if name in ["__inner_module__", "__jit_module__"]:
                return object.__getattribute__(self, name)
            
            nonlocal _original_methods, _trace_method_args
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
                    org_mod.load_state_dict(self.__jit_module__.state_dict(keep_vars=True))
                return getattr(self.__jit_module__, name)
            
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
    result = t.g(torch.rand(100, 4))
    print(result)
    t.add_mutable("mut_list", [torch.zeros(10), torch.ones(10)])
    t.add_mutable("mut_dict", {"a": torch.zeros(20), "b": torch.ones(20)})
    print(t.mut_list[0])
    print(t.mut_dict["b"])
    
    t = Test()
    fn = use_state(lambda: t.h, is_generator=True)
    trace_fn = torch.jit.trace(fn, (fn.init_state(), torch.ones(10, 1)), strict=False)
    
    def loop(init_state: Dict[str, torch.Tensor], init_x: torch.Tensor, n: int = 10):
        state = init_state
        ret = init_x
        rets: List[torch.Tensor] = []
        for _ in range(n):
            state, ret = trace_fn(state, ret)
            rets.append(state["self.sub_mod.buf"])
        return rets
    
    print(trace_fn.code)
    loop = torch.jit.trace(loop, (fn.init_state(), torch.rand(10, 2)), strict=False)
    print(loop.code)
    print(loop(fn.init_state(), torch.rand(10, 2)))
