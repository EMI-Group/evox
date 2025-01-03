import inspect
import warnings
from functools import wraps
from typing import Protocol, Callable, Optional, Union, Tuple, List, Dict, Any

import torch

from ..core.module import (
    tracing_or_using_state,
    trace_caching_state_context,
    UseStateFunc,
    _USE_STATE_NAME,
    _STATE_ARG_NAME,
)
from ..core import _vmap_fix


class MappedUseStateFunc(Protocol):

    def init_state(self, batch_size: int | None = None) -> Dict[str, torch.Tensor]:
        pass

    def __call__(
        self, state: Dict[str, torch.Tensor], *args, **kwargs
    ) -> Dict[str, torch.Tensor] | Tuple[Dict[str, torch.Tensor], Any]:
        pass


def vmap[
    T: Callable
](
    func: T | UseStateFunc,
    in_dims: Optional[int | Tuple[int, ...]] = 0,
    out_dims: Optional[int | Tuple[int, ...]] = 0,
    trace: bool = True,
    example_ndim: Tuple[int | None] | int = 1,
    example_shapes: Optional[Tuple[Tuple[int] | Any] | Tuple[int | Any]] = None,
    example_inputs: Optional[Tuple[torch.Tensor | Any]] = None,
    strict: bool = False,
    check_trace: bool = False,
    batched_state: Dict[str, torch.Tensor] | None = None,
    VMAP_DIM_CONST: int = 13,
) -> (T | MappedUseStateFunc):
    """Vectorize map the given function to its mapped version, see [`torch.vmap`](https://pytorch.org/docs/main/generated/torch.vmap.html) for more information.

    Args:
        func (Callable): The function to be mapped. See `torch.vmap`.
        in_dims (`int | Tuple[int, ...]`, optional): The inputs' batch dimensions. See `torch.vmap`. Defaults to 0.
        out_dims (`int | Tuple[int, ...]`, optional): The outputs' batch dimensions. See `torch.vmap`. Defaults to 0.
        trace (`bool`, optional): Whether to trace the mapped function with [`torch.jit.trace`](https://pytorch.org/docs/main/generated/torch.jit.trace.html) or simply use `torch.vmap. NOTICE: if `trace=False`, all of the following inputs related to tracing will be ignored.
        example_ndim (`Tuple[int | None] | int`): The `ndim` of the expected inputs of the batched function; thus, it must be at least 1. Giving a single integer means same `ndim` for all inputs. Defaults to 1.
        example_shapes (`Tuple[Tuple[int]  |  Any]  |  Tuple[int  |  Any]`, optional): The . Defaults to None.
        example_inputs (`Tuple[torch.Tensor  |  Any]`, optional): _description_. Defaults to None.
        strict (`bool`, optional): Strictly check the inputs or not. See `torch.jit.trace`. Defaults to False.
        check_trace (`bool`, optional): Check the traced function or not. See `torch.jit.trace`. Defaults to False.
        batched_state (`Dict[str, torch.Tensor]` or None, optional): The optional batched current state for a `use_state` wrapped function. If None, a new batched state will be returned for each call of `init_state(None)`. Ignored when `func` is not wrapped by `use_state`. Defaults to None.
        VMAP_DIM_CONST (`int`, optional): When tracing, the example inputs may be broadcasted with additional dimension(s) of size `VMAP_DIM_CONST`. Defaults to 13.

    Raises:
        NotImplementedError: If the function argument types are not supported

    Returns:
        `Callable`: The “batched” (vectorized mapped) version of `func`.
        If the given `func` is wrapped by `use_state`, the returned function will have a `init_state(batch_size: int) -> batched_state` or `init_state(None) -> batched_state`.
    """

    randomness = "same"  # prevent wrong batch tracing

    mapped = torch.vmap(func, in_dims, out_dims, randomness)
    # when tracing, do nothing
    if not trace or tracing_or_using_state():
        return mapped
    if hasattr(func, _USE_STATE_NAME):
        if batched_state is None:
            init_state = func.init_state()

            def _batched_init_state(batch_size: int | None = None):
                if batch_size is None:
                    batch_size = VMAP_DIM_CONST
                if isinstance(in_dims, tuple):
                    dim = in_dims[0]
                else:
                    dim = in_dims
                state = {}
                for k, v in init_state.items():
                    if isinstance(v, torch.nn.Parameter):
                        vv = v.data
                    else:
                        vv = v
                    vv = vv.unsqueeze(dim).expand(*v.shape[:dim], batch_size, *v.shape[dim:])
                    if isinstance(v, torch.nn.Parameter):
                        state[k] = torch.nn.Parameter(vv, requires_grad=v.requires_grad)
                    else:
                        state[k] = vv
                return state

            mapped.init_state = _batched_init_state
        else:
            mapped.init_state = lambda: batched_state
        object.__setattr__(mapped, _USE_STATE_NAME, True)
        return mapped

    # otherwise
    signature = inspect.signature(func).parameters
    args = []
    defaults = []
    annotations = []
    for k, v in signature.items():
        if v.kind in [
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ]:
            raise NotImplementedError(
                "Vector map of functions with variable or keyword-only arguments"
                + " are not supported when the example inputs are not provided"
            )
        args.append(k)
        defaults.append(None if v.default is inspect.Parameter.empty else v.default)
        annotations.append(None if v.annotation is inspect.Parameter.empty else v.annotation)
    if hasattr(func, "__self__"):
        args = args[1:]
        defaults = defaults[1:]
        annotations = annotations[1:]
    # check example shapes
    if example_shapes is not None:
        assert len(example_shapes) == len(
            args
        ), f"Expect example shapes to have size {
                len(args)}, got {len(example_shapes)}"
        example_ndim = tuple([None] * len(args))
    else:
        example_shapes = tuple([None] * len(args))
        if isinstance(example_ndim, int):
            assert (
                example_ndim >= 1
            ), f"Expect example ndim >= 1, got {
                example_ndim}"
            example_ndim = tuple([example_ndim] * len(args))
        else:
            assert len(example_ndim) == len(
                args
            ), f"Expect example ndim to have size {
                    len(args)}, got {len(example_ndim)}"
    # create example inputs
    example_inputs: List = [None] * len(args) if example_inputs is None else list(example_inputs)
    static_inputs = {}
    final_inputs = []
    for arg, default, annotation, input, shape, ndim in zip(
        args, defaults, annotations, example_inputs, example_shapes, example_ndim
    ):
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
                final_inputs.append(
                    torch.empty(()) if ndim is None else torch.empty(tuple([VMAP_DIM_CONST] * ndim))
                )
            else:
                try:
                    static_inputs[arg] = annotation()
                except Exception as e:
                    raise NotImplementedError(
                        f"Cannot create default value from annotation {annotation}", e
                    )
    # JIT
    final_inputs = tuple(final_inputs)
    if len(static_inputs) > 0:
        warnings.warn(
            f"The arguments {tuple(static_inputs.keys())} are not tensors or have default values,"
            + f" they will be set to {tuple(static_inputs.values())}"
            + " PERMANENTLY and shall be REMOVED during later invocation(s).",
            stacklevel=2,
        )
    wrapped = wraps(func)(lambda *args: mapped(*args, **static_inputs))
    jit_func = torch.jit.trace(wrapped, final_inputs, strict=strict, check_trace=check_trace)
    return jit_func


def jit[
    T: Callable
](
    func: T | UseStateFunc | MappedUseStateFunc,
    trace: bool = False,
    lazy: bool = False,
    example_inputs: Optional[Union[Tuple, Dict]] = None,
    strict: bool = False,
    check_trace: bool = False,
    is_generator: bool = False,
    no_cache: bool = False,
    return_dummy_output: bool = False,
) -> T:
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
        example_inputs (`tuple` or `dict`, optional): When `trace=True` and `lazy=False`, the example inputs must be provided immediately, otherwise ignored.
        strict (`bool`, optional): Strictly check the inputs or not. See [`torch.jit.trace`](https://pytorch.org/docs/main/generated/torch.jit.trace.html). Defaults to False.
        check_trace (`bool`, optional): Check the traced function or not. See [`torch.jit.trace`](https://pytorch.org/docs/main/generated/torch.jit.trace.html). Defaults to False.
        is_generator (`bool`, optional): Whether `func` is a generator or not. Defaults to False.
        no_cache (`bool`, optional): Whether to use `torch.jit.trace` directly (`no_cache=True`) or run the function to make it cache internals when `lazy=False`. Defaults to False. Has no effect when `trace=False` or `lazy=True`. This value must be set to `False` if the function contains a instant call to `torch.jit.trace` which will be used inside a `torch.jit.script` so that the JIT traced result shall be cached.
        return_dummy_output (`bool`, optional): Whether to return the dummy output or not. Defaults to False. Has no effect when `no_cache=True`.

    Returns:
        `Callable`: The JIT version of `func`
    """
    if is_generator:
        func = func()
    if isinstance(func, torch.jit.ScriptFunction):
        return func
    if not trace:
        return torch.jit.script_if_tracing(func) if lazy else torch.jit.script(func)
    elif not lazy:
        assert example_inputs is not None
        if isinstance(example_inputs, list):
            example_inputs = tuple(example_inputs)
        with _vmap_fix.use_batch_fixing():
            if isinstance(example_inputs, tuple):
                if not no_cache:
                    with trace_caching_state_context():
                        dummy_ret = func(*example_inputs)
                jit_func = torch.jit.trace(
                    func,
                    example_inputs,
                    strict=strict,
                    check_trace=check_trace,
                    _store_inputs=check_trace,
                )
            else:
                if not no_cache:
                    with trace_caching_state_context():
                        dummy_ret = func(**example_inputs)
                jit_func = torch.jit.trace(
                    func,
                    example_kwarg_inputs=example_inputs,
                    strict=strict,
                    check_trace=check_trace,
                    _store_inputs=check_trace,
                )
        if hasattr(func, _USE_STATE_NAME):
            func.set_state()  # reset global vars if using state
        return (jit_func, dummy_ret) if not no_cache and return_dummy_output else jit_func

    if hasattr(func, _USE_STATE_NAME):
        func_args = inspect.signature(func.__wrapped__).parameters.keys()
        func_args = list(func_args)
        func_args = [_STATE_ARG_NAME] + func_args
    else:
        func_args = inspect.signature(func).parameters.keys()
    func_args = tuple(func_args)
    jit_func = None

    @wraps(func)
    def jit_wrapper(*args, **kwargs):
        nonlocal jit_func, example_inputs
        with _vmap_fix.use_batch_fixing():
            if tracing_or_using_state():
                jit_func = func
                return func(*args, **kwargs)
            if not jit_func:
                # form positional inputs
                example_inputs = []
                arg_idx = 0
                for k in func_args:
                    if k in kwargs:
                        example_inputs.append(kwargs[k])
                    else:
                        assert arg_idx < len(args), (
                            f"Too few arguments, expected {len(func_args) - len(example_inputs)}"
                            + f" positional ones, got {len(args)}"
                        )
                        example_inputs.append(args[arg_idx])
                        arg_idx += 1
                # clone tensor inputs to remove influences of in-place operations
                example_inputs = tuple(example_inputs)
                leaves, tree_spec = _vmap_fix.tree_flatten(example_inputs)
                leaves = [l.clone() if isinstance(l, torch.Tensor) else l for l in leaves]
                example_inputs = _vmap_fix.tree_unflatten(leaves, tree_spec)
                # JIT trace
                jit_func = torch.jit.trace(
                    func,
                    example_inputs,
                    strict=strict,
                    check_trace=check_trace,
                    _store_inputs=check_trace,
                )
                # reset global vars if using state
                if hasattr(func, _USE_STATE_NAME):
                    func.set_state()
            return jit_func(*args, **kwargs)

    return jit_wrapper
