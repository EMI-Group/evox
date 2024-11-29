import inspect
import types
import warnings
from functools import wraps
from typing import Callable, Optional, Sequence, Union, Tuple, List, Dict, Any

import torch
from torch import nn


def vmap(func: Callable,
         in_dims: Optional[int | Tuple[int, ...]] = 0,
         out_dims: Optional[int | Tuple[int, ...]] = 0,
         randomness: str = "different",
         strict: bool = False,
         example_ndim: Tuple[int | None] | int = 1,
         example_shapes: Optional[Tuple[Tuple[int] | Any] | Tuple[int | Any]] = None,
         example_inputs: Optional[Tuple[torch.Tensor | Any]] = None) -> Callable:
    """Vectorize map the given function to its mapped version, see [`torch.vmap`](https://pytorch.org/docs/main/generated/torch.vmap.html) for more information.

    Args:
        func (Callable): The function to be mapped. See `torch.vmap`.
        in_dims (`int | Tuple[int, ...]`, optional): The inputs' batch dimensions. See `torch.vmap`. Defaults to 0.
        out_dims (`int | Tuple[int, ...]`, optional): The outputs' batch dimensions. See `torch.vmap`. Defaults to 0.
        randomness (str, optional): The randomness in this vmap. See `torch.vmap`. Defaults to "different".
        strict (bool, optional): Strictly check the inputs or not. See [`torch.jit.trace`](https://pytorch.org/docs/main/generated/torch.jit.trace.html). Defaults to False.
        example_ndim (`Tuple[int | None] | int`): The `ndim` of the expected inputs of the batched function; thus, it must be at least 1. Giving a single integer means same `ndim` for all inputs. Defaults to 1.
        example_shapes (`Tuple[Tuple[int]  |  Any]  |  Tuple[int  |  Any]`, optional): The . Defaults to None.
        example_inputs (`Tuple[torch.Tensor  |  Any]`, optional): _description_. Defaults to None.

    Raises:
        NotImplementedError: If the function argument types are not supported

    Returns:
        `Callable`: The “batched” (vectorized mapped) version of `func`.
    """
    
    VMAP_DIM_CONST = 13
    
    mapped = torch.vmap(func, in_dims, out_dims, randomness)
    # when tracing, do nothing
    if torch.jit.is_tracing():
        return mapped
    
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
    example_inputs: List = [None] * len(args) if example_inputs is None else list(example_inputs)
    static_inputs = {}
    final_inputs = []
    for arg, default, annotation, input, shape, ndim in zip(args, defaults, annotations, example_inputs, example_shapes,
                                                            example_ndim):
        if input is not None:
            final_inputs.append(input)
        elif shape is not None and (isinstance(shape, int) or isinstance(shape, tuple)):
            final_inputs.append(torch.empty(shape))
        elif default is not None:
            # if isinstance(default, torch.Tensor):
            #     static_inputs[arg] = default
            # else:
            #     static_inputs[arg] = default
            static_inputs[arg] = default
        else:
            if annotation == torch.Tensor:
                final_inputs.append(torch.empty(()) if ndim is None else torch.empty(tuple([VMAP_DIM_CONST] * ndim)))
            else:
                try:
                    static_inputs[arg] = annotation()
                except Exception as e:
                    raise NotImplementedError(f"Cannot create default value from annotation {annotation}", e)
    # JIT
    final_inputs = tuple(final_inputs)
    if len(static_inputs) > 0:
        warnings.warn(f"The arguments {tuple(static_inputs.keys())} are not tensors or have default values," +
                      f" they will be set to {tuple(static_inputs.items())}" +
                      " PERMANENTLY and shall be REMOVED during later invocation(s).",
                      stacklevel=2)
    
    def wrapper(*args):
        return mapped(*args, **static_inputs)
    
    jit_func = torch.jit.trace(wrapper, final_inputs, strict=strict)
    return jit_func


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


_TRACE_WRAP_NAME = "__trace_wrapped__"
_SELF_STATE_DICT_NAME = "__self_state_dict__"
global _TORCHSCRIPT_MODIFIER
_TORCHSCRIPT_MODIFIER = "_torchscript_modifier"


def trace_impl(target: Callable):
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


def _get_inner_state_func(state_wrapper, func):
    func = use_state(func)
    
    @wraps(func)
    def _inner_state_func(*args, **kwargs):
        _new_state_dict, ret = func(state_wrapper.__state_dict__, *args, **kwargs)
        state_wrapper.__state_dict__.update(_new_state_dict)
        return ret
    
    return _inner_state_func


def _is_bound_method(value):
    return callable(value) and hasattr(value, "__self__") and not hasattr(value, "__use_state__") and \
            not (value.__name__.startswith("__") and value.__name__.endswith("__"))


def use_state(func: Callable) -> Callable:
    if not _is_bound_method(func):
        return func
    
    parameters = inspect.signature(func).parameters
    assert _SELF_STATE_DICT_NAME not in parameters, \
        f"The function parameter cannot be named as internal preserved '{_SELF_STATE_DICT_NAME}'"
    func_self = func.__self__
    unbound_func = func.__func__
    
    class SelfStateWrapper:
        
        def __init__(self, state_dict: Dict[str, torch.Tensor]):
            self.__state_dict__ = state_dict
        
        def __getitem__(self, key):
            return func_self.__getitem__(key)
        
        def __setitem__(self, value, key):
            func_self.__setitem__(value, key)
        
        def __setattr__(self, name, value):
            if name != "__state_dict__":
                if name in self.__state_dict__:
                    self.__state_dict__[name] = value
                else:
                    setattr(func_self, name, value)
            else:
                object.__setattr__(self, name, value)
        
        def __getattribute__(self, name):
            if name == "__state_dict__":
                return object.__getattribute__(self, name)
            # has cache
            if name in self.__state_dict__:
                return self.__state_dict__[name]
            value = getattr(func_self, name)
            # sub module
            if isinstance(value, nn.Module):
                for name, method in value.__dict__.items():
                    if not _is_bound_method(method):
                        continue
                    # TODO: not correct
                    setattr(value, name, use_state(method))
                # TODO: return what?
            # function
            if _is_bound_method(value):
                value = _get_inner_state_func(self, value)
                self.__state_dict__[name] = value
                return value
            # else
            return value
    
    @wraps(func)
    def wrapper(__self_state_dict__, *args, **kwargs):
        state = SelfStateWrapper(__self_state_dict__)
        ret = unbound_func(state, *args, **kwargs)
        return state.__state_dict__, ret
    
    wrapper.__use_state__ = True
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
    
    class WrappedModuleType(type):
        
        def __getattr__(cls_new, name):
            return getattr(cls, name)
        
        def __setattr__(cls_new, name, value):
            return setattr(cls, name, value)
    
    @wraps(cls, updated=())
    class WrappedModule(metaclass=WrappedModuleType):
        
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
            if not trace and not torch.jit.is_tracing():
                if jit_mod is None:
                    self.__jit_module__ = torch.jit.script(org_mod)
                return self.__jit_module__.__getattr__(name)
            
            # deal with pure trace
            func = _original_methods[name]
            func = use_state(func)
            
            @wraps(func)
            def method_wrapper(*args, **kwargs):
                if torch.jit.is_tracing():
                    if not trace and name in _trace_correspond_methods:
                        if hasattr(_trace_correspond_methods[name], "__self__"):
                            bounded_trace_target_func = _trace_correspond_methods[name]
                        else:
                            bounded_trace_target_func = types.MethodType(_trace_correspond_methods[name], org_mod)
                            _trace_correspond_methods[name] = bounded_trace_target_func
                        func = use_state(bounded_trace_target_func)
                    return func(org_mod.state_dict(), *args, **kwargs)
                if name not in _trace_method_args:
                    _trace_method_args[name] = {
                        _SELF_STATE_DICT_NAME: org_mod.state_dict(),
                        **dict(zip(_method_argnames[name], args)),
                        **kwargs
                    }
                    traced_func = torch.jit.trace(func, _trace_method_args, example_inputs_is_kwarg=True)
                return traced_func(*args, **kwargs)
            
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
    from functools import partial
    
    @partial(vmap, example_ndim=2)
    def _single_eval(x: torch.Tensor, p: float = 2.0, q: torch.Tensor = torch.as_tensor(range(2))):
        return (x**p).sum() * q.sum()
    
    print(_single_eval(2 * torch.ones(10, 2)))
    print(torch.jit.script(_single_eval)(2 * torch.ones(10, 2)))
    
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
            x = torch.where(q.flatten()[0] > self.threshold, torch.sin(q), torch.tan(q))
            x += self.g(x)
            self.sub_mod.buf = x.sum()
            return x * x.shape[1]
        
        def g(self, p: torch.Tensor) -> torch.Tensor:
            x = torch.cos(p)
            return x * p.shape[0]
    
    # t = Test()
    # print(t.h.inlined_graph)
    # result = t.g(torch.randn(100, 4))
    # print(result)
    # t.add_mutable("mut_list", [torch.zeros(10), torch.ones(10)])
    # t.add_mutable("mut_dict", {"a": torch.zeros(20), "b": torch.ones(20)})
    # print(t.mut_list[0])
    # print(t.mut_dict["b"])
    
    t = Test()
    def fn(q: torch.Tensor):
        return t.h(q)
    
    trace_fn = torch.jit.trace(fn, torch.empty(10, 1))
    print(trace_fn.inlined_graph)
    