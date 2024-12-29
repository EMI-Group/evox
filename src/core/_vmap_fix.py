from functools import wraps
from typing import Any, Callable, List, Tuple

import torch
import torch._C._functorch as _functorch
from torch._C._functorch import (
    _add_batch_dim as add_batch_dim,
    maybe_get_bdim as get_batch_dim,
    maybe_get_level as get_level,
    is_batchedtensor as is_batched_tensor,
    _unwrap_batched as unwrap_batched,
    _vmap_decrement_nesting as vmap_decrement_nesting,
    _vmap_increment_nesting as vmap_increment_nesting,
)

if "maybe_current_level" not in _functorch.__dict__:

    def current_level() -> int | None:
        try:
            return _functorch.current_level()
        except:
            return None

else:
    current_level = _functorch.maybe_current_level

from torch.utils._pytree import tree_flatten, tree_unflatten
from torch import nn

if "Buffer" not in nn.__dict__:
    nn.Buffer = lambda x: x


def _transform_in_dim(
    in_dim: int | Tuple[int, ...], batched: torch.Tensor, original: torch.Tensor
) -> int | Tuple[int, ...] | None:
    if in_dim is None:
        return None
    if not isinstance(in_dim, tuple):
        in_dim = (in_dim,)
    shape = original.size()
    batch_dims_shape = tuple(s for i, s in enumerate(shape) if i in in_dim)
    shape = tuple(s for i, s in enumerate(shape) if i not in in_dim)
    batched.size = lambda i=None: shape if i is None else shape[i]
    batched.__dict__["shape"] = shape
    return batch_dims_shape if len(batch_dims_shape) > 1 else batch_dims_shape[0]


import torch._functorch.vmap as vmap

global __vmap_batch_sizes__
__vmap_batch_sizes__ = []


def _create_batched_inputs(
    flat_in_dims: List[Any], flat_args: List[Any], vmap_level: int, args_spec
) -> Tuple:
    # See NOTE [Ignored _remove_batch_dim, _add_batch_dim]
    batched_inputs = [
        arg if in_dim is None else add_batch_dim(arg, in_dim, vmap_level)
        for in_dim, arg in zip(flat_in_dims, flat_args)
    ]
    batch_size = None
    for batched, arg, in_dim in zip(batched_inputs, flat_args, flat_in_dims):
        if isinstance(batched, torch.Tensor):
            bs = _transform_in_dim(in_dim, batched, arg)
            if bs is not None:
                batch_size = bs
    global __vmap_batch_sizes__
    __vmap_batch_sizes__.append(batch_size)
    return tree_unflatten(batched_inputs, args_spec)


def _unwrap_batched(
    batched_outputs: torch.Tensor | Tuple[torch.Tensor, ...],
    out_dims: int | Tuple[int, ...],
    vmap_level: int,
    batch_size: int,
    func: Callable,
) -> Tuple:
    flat_batched_outputs, output_spec = tree_flatten(batched_outputs)

    def incompatible_error():
        raise ValueError(
            f"vmap({vmap._get_name(func)}, ..., out_dims={out_dims})(<inputs>): "
            f"out_dims is not compatible with the structure of `outputs`. "
            f"out_dims has structure {tree_flatten(out_dims)[1]} but outputs "
            f"has structure {output_spec}."
        )

    if isinstance(batched_outputs, torch.Tensor):
        # Some weird edge case requires us to spell out the following
        # see test_out_dims_edge_case
        if isinstance(out_dims, int):
            flat_out_dims = [out_dims]
        elif isinstance(out_dims, tuple) and len(out_dims) == 1:
            flat_out_dims = out_dims
        elif out_dims is None:
            flat_out_dims = [out_dims]
        else:
            incompatible_error()
    else:
        flat_out_dims = vmap._broadcast_to_and_flatten(out_dims, output_spec)
        if flat_out_dims is None:
            incompatible_error()

    flat_outputs = [
        vmap._maybe_remove_batch_dim(
            vmap._get_name(func), batched_output, vmap_level, batch_size, out_dim
        )
        for batched_output, out_dim in zip(flat_batched_outputs, flat_out_dims)
    ]
    global __vmap_batch_sizes__
    __vmap_batch_sizes__.pop()
    return tree_unflatten(flat_outputs, output_spec)


vmap._create_batched_inputs = _create_batched_inputs
vmap._unwrap_batched = _unwrap_batched


def unwrap_batch_tensor(tensor: torch.Tensor):
    level = get_level(tensor)
    batch_dims = []
    batch_sizes = []
    while level >= 1:
        batch_dim = get_batch_dim(tensor)
        tensor, _ = unwrap_batched(tensor, level)
        batch_dims.append(batch_dim)
        batch_sizes.append(tensor.size(batch_dim))
        level -= 1
    batch_dims = tuple(batch_dims[::-1])
    batch_sizes = tuple(batch_sizes[::-1])
    return tensor, batch_dims, batch_sizes

def align_vmap_tensor(value: Any, current_value: Any | None):
    """
    Aligns a tensor with the batching dimensions of a current batched tensor.

    This function adjusts the input tensor `value` to match the batch dimensions
    of `current_value`, which is assumed to be a batched tensor. If `value` is
    already a batched tensor or `current_value` is not a batched tensor, it
    returns `value` unchanged.

    Args:
        value (Any): The tensor to be aligned. If not a `torch.Tensor`, it is
                     returned unchanged.
        current_value (Any | None): The reference batched tensor. If `None` or
                                    not a batched tensor, `value` is returned
                                    unchanged.

    Returns:
        `torch.Tensor`: The input `value` aligned with the batch dimensions of
                      `current_value`, if applicable.
    """

    if not isinstance(value, torch.Tensor):
        return value
    if is_batched_tensor(value):
        return value
    if current_value is None or not is_batched_tensor(current_value):
        return value
    # level = get_level(current_value)
    # base_value = current_value
    # batch_dims = []
    # batch_sizes = []
    # while level >= 1:
    #     batch_dim = get_batch_dim(base_value)
    #     base_value, _ = unwrap_batched(base_value, level)
    #     batch_dims.append(batch_dim)
    #     batch_sizes.append(base_value.size(batch_dim))
    #     level -= 1
    # batch_dims = tuple(batch_dims[::-1])
    # batch_sizes = tuple(batch_sizes[::-1])
    value, batch_dims, batch_sizes = unwrap_batch_tensor(current_value)
    for dim, size in zip(batch_dims, batch_sizes):
        value = value.unsqueeze(dim).expand(*value.shape[:dim], size, *value.shape[dim:])
    expand_value = value
    for level, dim in enumerate(batch_dims, 1):
        value = add_batch_dim(value, dim, level)
    _transform_in_dim(batch_dims, value, expand_value)
    return value


def wrap_vmap_inputs[T: Callable](func: T) -> T:
    """
    Wraps a function to adjust its input tensors for vmap compatibility.

    This decorator modifies the input tensors of the wrapped function to align
    with the batching dimensions expected by `vmap`. It ensures that tensors
    which are already batched or not tensors remain unchanged, while tensors
    that need to be batched are transformed accordingly.

    Args:
        func (`Callable`): The function to be wrapped, which will have its input
                         tensors adjusted for vmap.

    Returns:
        `Callable`: A wrapped version of `func` that ensures its input tensors
                  are compatible with vmap's batching requirements.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        flat_args, flat_spec = tree_flatten((args, kwargs))
        for arg in flat_args:
            if not isinstance(arg, torch.Tensor):
                continue
            if not is_batched_tensor(arg):
                continue
            # level = get_level(arg)
            # base_arg = arg
            # batch_dims = []
            # while level >= 1:
            #     batch_dim = get_batch_dim(base_arg)
            #     base_arg, _ = unwrap_batched(base_arg, level)
            #     batch_dims.append(batch_dim)
            #     level -= 1
            # batch_dims = tuple(batch_dims[::-1])
            unwrap_arg, batch_dims, _ = unwrap_batch_tensor(arg)
            _transform_in_dim(batch_dims, arg, unwrap_arg)
        args, kwargs = tree_unflatten(flat_args, flat_spec)
        return func(*args, **kwargs)

    return wrapper


def batched_random(rand_func: Callable, *size: Tuple[int | torch.SymInt], **kwargs) -> torch.Tensor:
    """
    Generate a batched tensor of random values.

    Given a random function (e.g. [`torch.randn`](https://pytorch.org/docs/stable/generated/torch.randn.html),
    [`torch.rand`](https://pytorch.org/docs/stable/generated/torch.rand.html), etc.) and its size arguments, this function
    generates a batched tensor of random values by applying the given function to
    the size extended with the current vmap batch size.

    Args:
        rand_func (`Callable`): A function that generates a tensor of random values.
        *size (`Tuple[int | torch.SymInt]`): The size arguments to the given function.
        **kwargs: The keyword arguments to the given function.

    Returns:
        torch.Tensor: The batched tensor of random values.
        
    ## Usage:
    ```
    rand1 = batched_random(torch.rand, 2, 3, device=device)
    rand2 = batched_random(torch.randn, 4, device=device, dtype=torch.float32)
    rand3 = batched_random(torch.randint, 5, 6, low=0, high=10, device=device, dtype=torch.float32)
    ```
    """
    level = current_level()
    if level is None or level <= 0:
        return rand_func(size=size, **kwargs)
    # else
    global __vmap_batch_sizes__
    size = tuple(__vmap_batch_sizes__) + size
    num_levels = len(__vmap_batch_sizes__)
    rand_values = rand_func(size=size, **kwargs)
    batched_rand_values = rand_values
    for level in range(1, num_levels + 1):
        batched_rand_values = add_batch_dim(batched_rand_values, 0, level)
    _transform_in_dim(tuple(range(num_levels)), batched_rand_values, rand_values)
    return batched_rand_values
