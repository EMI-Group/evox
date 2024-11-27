import ast
import inspect
import warnings
from functools import wraps
from typing import Callable, Optional, Sequence, Union, Tuple, List, Dict, Any

import torch
from torch import nn


def vmap(func: Callable,
         in_dims: Optional[int | Tuple[int, ...]] = 0,
         out_dims: Optional[int | Tuple[int, ...]] = 0,
         randomness: str = "error",
         strict: bool = False,
         example_ndim: Tuple[int | None] | int = 1,
         example_shapes: Optional[Tuple[Tuple[int] | Any] | Tuple[int | Any]] = None,
         example_inputs: Optional[Tuple | Dict] = None) -> Callable:
    mapped = torch.vmap(func, in_dims, out_dims, randomness)
    # when tracing, do nothing
    if torch.jit.is_tracing():
        return mapped
    # when example inputs are provided
    if example_inputs is not None:
        if isinstance(example_inputs, dict):
            return torch.jit.trace(mapped, example_kwarg_inputs=example_inputs, strict=strict)
        else:
            return torch.jit.trace(mapped, example_inputs=example_inputs, strict=strict)
    # otherwise
    signature = inspect.signature(func).parameters
    args = []
    defaults = []
    annotations = []
    for k, v in signature.items():
        if v.kind in [inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
            raise NotImplementedError("Vector map of functions with variable or keyword-only arguments" +
                                      " are not supported when the example inputs are not provided")
        args.append(k)
        defaults.append(None if v.default is inspect.Parameter.empty else v.default)
        annotations.append(None if v.annotation is inspect.Parameter.empty else v.annotation)
    if hasattr(func, "__self__"):
        args = args[1:]
        defaults = defaults[1:]
        annotations = annotations[1:]
    # check example shapes
    if example_shapes is not None:
        assert len(example_shapes) == len(args), f"Expect example shapes to have size {len(args)}, got {len(example_shapes)}"
        example_ndim = tuple([None] * len(args))
    else:
        example_shapes = tuple([None] * len(args))
        if isinstance(example_ndim, int):
            assert example_ndim >= 1, f"Expect example ndim >= 1, got {example_ndim}"
            example_ndim = tuple([example_ndim] * len(args))
        else:
            assert len(example_ndim) == len(args), f"Expect example ndim to have size {len(args)}, got {len(example_ndim)}"
    # create example inputs
    example_inputs: List = []
    for arg, default, annotation, shape, ndim in zip(args, defaults, annotations, example_shapes, example_ndim):
        if shape is not None and (isinstance(shape, int) or isinstance(shape, tuple)):
            example_inputs.append(torch.empty(shape))
            continue
        if default is not None:
            example_inputs.append(default)
            continue
        if ndim is not None:
            assert annotation is not None and annotation == torch.Tensor, \
                f"Vector map of functions with argument {arg} of non-tensor type at compilation is not supported"
            example_inputs.append(torch.empty(tuple([13] * ndim)))
            continue
        if annotation is not None:
            try:
                example_inputs.append(annotation())
            except Exception as e:
                raise TypeError(f"Cannot create default value from annotation {annotation}", e)
    args_to_tensor = []
    for i, arg in enumerate(args):
        try:
            example_inputs[i] = torch.as_tensor(example_inputs[i])
            args_to_tensor.append(True)
        except Exception as e:
            warnings.warn(f"Cannot convert argument {arg} to tensor, it will be treated as STATIC argument" +
                          " and shall be REMOVED during invocation.",
                          source=e)
            args_to_tensor.append(False)
    example_inputs = tuple(example_inputs)
    args_to_tensor = tuple(args_to_tensor)
    # JIT
    jit_func = torch.jit.trace(mapped, example_inputs, strict=strict)
    def wrapped(x: torch.Tensor, p: float = 2.0):
        p = torch.as_tensor(p)
        return jit_func(x, p)
    return wrapped
    
#     wrapped_func = \
# f"""def wrapped({", ".join(a + ": " + ("" if t.__module__ == "builtins" else t.__module__ + ".") + t.__name__ for a, t in zip(args, annotations))}):
#     {"; ".join((a + " = torch.as_tensor(" + a + ")") if t \
#                 else ("") for a, t in zip(args, args_to_tensor))}
#     return jit_func({", ".join(args)})
# """
#     g = globals()
#     g["jit_func"] = jit_func
#     exec(wrapped_func, g, locals())
#     return locals()["wrapped"]


def jit(func: Callable,
        trace: bool = False,
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
        trace (`bool`, optional): Whether using `torch.jit.trace` or `torch.jit.script` to JIT. Defaults to False.
        lazy (`bool`, optional): Whether JIT lazily or immediately. Defaults to False.
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
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal jit_func
        if torch.jit.is_tracing():
            jit_func = func
            return func(*args, **kwargs)
        if not jit_func:
            jit_func = torch.jit.trace(func, example_kwarg_inputs={**dict(zip(args_specs.args, args)), **kwargs})
        return jit_func(*args, **kwargs)
    
    return wrapper


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
    _original_methods = {}
    _method_argnames = {}
    _trace_method_args = {}
    _jit_module = None
    for name, method in cls.__dict__.items():
        if callable(method):
            args_specs = inspect.getfullargspec(method)
            if len(args_specs.args) == 0 or args_specs.args[0] != "self":
                continue
            if name.startswith("__") and name.endswith("__"):
                continue
            if not trace:
                torch.jit.export(method)
            _original_methods[name] = method
            _method_argnames[name] = args_specs.args[1:]
    
    class WrappedModuleType(type):
        
        def __getattr__(cls_new, name):
            return getattr(cls, name)
        
        def __setattr__(cls_new, name, value):
            return setattr(cls, name, value)
    
    @wraps(cls, updated=())
    class WrappedModule(metaclass=WrappedModuleType):
        
        def __init__(self, *args, **kwargs):
            self.__inner_module__ = cls(*args, **kwargs)
        
        def __str__(self) -> str:
            return object.__str__(self.__inner_module__)
        
        def __repr__(self) -> str:
            return object.__repr__(self.__inner_module__)
        
        def __hash__(self) -> int:
            return object.__hash__(self.__inner_module__)
        
        def __format__(self, format_spec: str) -> str:
            return object.__format__(self.__inner_module__, format_spec)
        
        def __getitem__(self, key):
            return self.__inner_module__.__getitem__(key)
        
        def __setitem__(self, value, key):
            self.__inner_module__.__setitem__(value, key)
        
        def __setattr__(self, name, value):
            if name != "__inner_module__":
                setattr(self.__inner_module__, name, value)
            else:
                object.__setattr__(self, name, value)
        
        def __delattr__(self, name, value):
            if name != "__inner_module__":
                delattr(self.__inner_module__, name, value)
            else:
                object.__delattr__(self, name, value)
        
        def __getattr__(self, name):
            nonlocal _original_methods, _trace_method_args, _jit_module
            if name == "__inner_module__":
                return object.__getattribute__(self, name)
            if name not in _original_methods:
                return getattr(self.__inner_module__, name)
            
            # deal with script
            if not trace:
                if _jit_module is None:
                    _jit_module = torch.jit.script(self.__inner_module__)
                return _jit_module.__getattr__(name)
            
            # deal with trace
            attr = object.__getattribute__(self.__inner_module__, name)
            
            @wraps(attr)
            def method_wrapper(*args, **kwargs):
                nonlocal _jit_module
                if torch.jit.is_tracing():
                    return attr(*args, **kwargs)
                if name not in _trace_method_args:
                    _trace_method_args[name] = {**dict(zip(_method_argnames[name], args)), **kwargs}
                    _jit_module = torch.jit.trace_module(self.__inner_module__,
                                                         _trace_method_args,
                                                         example_inputs_is_kwarg=True)
                return _jit_module.__getattr__(name)(*args, **kwargs)
            
            return method_wrapper
    
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


if __name__ == "__main__":
    
    # @torch.jit.script
    def _single_eval(x: torch.Tensor, p: float = 2.0):
        return (x**p).sum()
    
    mapped = torch.vmap(_single_eval)
    inspect.getsource(mapped)
    _multi_eval = torch.jit.script(mapped)
    print(_multi_eval(2 * torch.ones(10, 2)))
    print(_multi_eval(2 * torch.ones(10, 2), p=3.0))
    
    # _multi_eval = vmap(_single_eval, in_dims=(0, None), example_ndim=2)
    # _multi_eval = torch.jit.script(_multi_eval)
    # print(_multi_eval(2 * torch.ones(10, 2), 3.0))
    # print(_multi_eval(2 * torch.ones(10, 2)))
    
    # @jit_class
    # class Test(ModuleBase):
    
    #     def __init__(self, threshold=0.5):
    #         super().__init__()
    #         self.threshold = threshold
    
    #     def h(self, q: torch.Tensor) -> torch.Tensor:
    #         if q.flatten()[0] > self.threshold:
    #             x = torch.sin(q)
    #         else:
    #             x = torch.tan(q)
    #         return x * x.shape[1]
    
    #     def g(self, p: torch.Tensor) -> torch.Tensor:
    #         x = torch.cos(p)
    #         return x * p.shape[0]
    
    # t = Test()
    # print(t.h.inlined_graph)
    # result = t.g(torch.randn(100, 4))
    # print(result)
    # t.add_mutable("mut_list", [torch.zeros(10), torch.ones(10)])
    # t.add_mutable("mut_dict", {"a": torch.zeros(20), "b": torch.ones(20)})
    # print(t.mut_list[0])
    # print(t.mut_dict["b"])
