from functools import wraps
from typing import Callable, Dict, Optional, TypeVar, Union

import torch
import torch.nn as nn

from ..core import _vmap_fix

_WRAPPING_MODULE_NAME = "__wrapping_module__"

ParameterT = TypeVar("ParameterT", torch.Tensor, int, float, complex)


def Parameter(
    value: ParameterT,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> ParameterT:
    """Wraps a value as parameter with `requires_grad=False`.
    This is often used to label a value in an algorithm as a hyperparameter that can be identified by the `HPOProblemWrapper`.

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


def Mutable(
    value: torch.Tensor, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Wraps a value as a mutable tensor.
    This is often used to label a value in an algorithm as a mutable tensor that may changes during iteration(s).

    :param value: The value to be wrapped.
    :param dtype: The dtype of the tensor. Defaults to None.
    :param device: The device of the tensor. Defaults to None.

    :return: The wrapped tensor.
    """
    return nn.Buffer(value.to(dtype=dtype, device=device))


class ModuleBase(nn.Module):
    """
    The base module for all algorithms, problems, and workflows in the library.

    :note: To prevent ambiguity, `ModuleBase.eval()` is disabled.
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


# We still need a fix for the vmap
# related issue: https://github.com/pytorch/pytorch/issues/124423
class TransformGetSetItemToIndex(TorchFunctionMode):
    # This is needed since we want to support calling
    # A[idx] or A[idx] += b, where idx is a scalar tensor.
    # When idx is a scalar tensor, Torch implicitly convert it to a python
    # scalar and create a view of A.
    # Workaround: We convert the scalar tensor to a 1D tensor with one element.
    # That is, we convert A[idx] to A[idx[None]][0], A[idx] += 1 to A[idx[None]] += 1.
    # This is a temporary solution until the issue is fixed in PyTorch.
    def __torch_function__(self, func, types, args, kwargs=None):
        if func == torch.Tensor.__getitem__:
            x, index = args
            if isinstance(index, torch.Tensor) and index.ndim == 0:
                return func(x, index[None], **(kwargs or {}))[0]
                # return torch.index_select(x, 0, index)
        elif func == torch.Tensor.__setitem__:
            x, index, value = args
            if isinstance(index, torch.Tensor) and index.ndim == 0:
                return func(x, index[None], value, **(kwargs or {}))

        return func(*args, **(kwargs or {}))


@wraps(torch.compile)
def compile(*args, **kwargs) -> Callable:
    """Fix the `torch.compile`'s issue with __getitem__ and __setitem__
    that recognizes scalar indexes as `.item()` and causes graph breaks.
    Related issue: https://github.com/pytorch/pytorch/issues/124423.
    """

    with TransformGetSetItemToIndex():
        compiled = torch.compile(*args, **kwargs)

    def wrapper(*args, **kwargs):
        with TransformGetSetItemToIndex():
            return compiled(*args, **kwargs)

    wrapper.__wrapped__ = compiled
    return wrapper


@wraps(torch.vmap)
def vmap(*args, **kwargs) -> Callable:
    """Fix the `torch.vmap`'s issue with __getitem__ and __setitem__.
    Related issue: https://github.com/pytorch/pytorch/issues/124423.
    """
    A context manager to set the value of `using_state` temporarily.

    When entering the context, the value of `using_state` is set to `new_use_state` and a token is obtained.
    When exiting the context, the value of `using_state` is reset to its previous value.

    :param new_use_state: The new value of `using_state`. Defaults to True.

    return wrapper


class _ReplaceForwardModule(nn.Module):
    def __init__(self, module: nn.Module, new_forward: Callable):
        super().__init__()
        self._inner_module = module
        self.new_forward = new_forward

    def forward(self, *args, **kwargs):
        return self.new_forward(self._inner_module, *args, **kwargs)


def use_state(stateful_func: Union[Callable, nn.Module]) -> Callable:
    """Transform a `torch.nn.Module`'s method or an `torch.nn.Module` into a stateful function.
    When using `torch.nn.Module`, the stateful version of the default `forward` method will be created.
    The stateful function will have a signature of `fn(params_and_buffers, *args, **kwargs) -> params_and_buffers | Tuple[params_and_buffers, <original_returns>]]`.

    ## Examples

    ```python
    from evox import use_state, vmap
    workflow = ... # define your workflow
    stateful_step = use_state(workflow.step)
    vmap_stateful_step = vmap(stateful_step)
    batch_state = torch.func.stack_module_states([workflow] * 3)
    new_batch_state = vmap_stateful_step(batch_state)
    ```
    """
    if not isinstance(stateful_func, torch.nn.Module):
        module: torch.nn.Module = stateful_func.__self__
        assert isinstance(module, torch.nn.Module), (
            "`stateful_func` must be a `torch.nn.Module` or a method of a `torch.nn.Module`"
        )
        new_forward = stateful_func.__func__
    else:
        module = stateful_func
        new_forward = module.forward.__func__
    module = _ReplaceForwardModule(module, new_forward)

    def wrapper(params_and_buffers: Dict[str, torch.Tensor], *args, **kwargs):
        params_and_buffers = {("_inner_module." + k): v for k, v in params_and_buffers.items()}
        output = torch.func.functional_call(module, params_and_buffers, *args, **kwargs)
        params_and_buffers = {k[len("_inner_module."):]: v for k, v in params_and_buffers.items()}
        if output is None:
            return params_and_buffers
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
