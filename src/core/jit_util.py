import inspect
import warnings
from functools import wraps
from typing import Callable, Optional, Union, Tuple, List, Dict, Any

import torch

from module import tracing_or_using_state


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
    if tracing_or_using_state():
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
        lazy: bool = False,
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
        if tracing_or_using_state():
            jit_func = func
            return func(*args, **kwargs)
        if not jit_func:
            jit_func = torch.jit.trace(func, example_kwarg_inputs={**dict(zip(args_specs.args, args)), **kwargs})
        return jit_func(*args, **kwargs)
    
    return wrapper


if __name__ == "__main__":
    from functools import partial
    
    @partial(vmap, example_ndim=2)
    def _single_eval(x: torch.Tensor, p: float = 2.0, q: torch.Tensor = torch.as_tensor(range(2))):
        return (x**p).sum() * q.sum()
    
    print(_single_eval(2 * torch.ones(10, 2)))
    print(jit(_single_eval)(2 * torch.ones(10, 2)))
    print(jit(_single_eval, trace=True, lazy=True)(2 * torch.ones(10, 2)))
