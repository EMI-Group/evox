import contextlib
from functools import wraps
from typing import Any, Callable, List, Sequence, Tuple

import torch
from torch._C._functorch import (
    _add_batch_dim as add_batch_dim,
    maybe_get_bdim as get_batch_dim,
    maybe_get_level as get_level,
    maybe_current_level as current_level,
    is_batchedtensor as is_batched_tensor,
    get_unwrapped,
    _unwrap_batched as unwrap_batched,
    _vmap_increment_nesting as vmap_increment_nesting,
    _vmap_decrement_nesting as vmap_decrement_nesting,
)

from torch.utils._pytree import tree_flatten, tree_unflatten


def _transform_in_dim(
    in_dim: int | Tuple[int, ...], batched: torch.Tensor, original: torch.Tensor
) -> int | Tuple[int, ...] | None:
    if in_dim is None:
        return None
    if not isinstance(in_dim, tuple):
        in_dim = (in_dim,)
    shape = original.size()
    batch_shape = tuple(s for i, s in enumerate(shape) if i in in_dim)
    shape = tuple(s for i, s in enumerate(shape) if i not in in_dim)
    batched.size = lambda i=None: shape if i is None else shape[i]
    return batch_shape if len(batch_shape) > 1 else batch_shape[0]


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


def align_vmap_tensor(value: Any, current_value: Any | None):
    if not isinstance(value, torch.Tensor):
        return value
    if is_batched_tensor(value):
        return value
    if current_value is None or not is_batched_tensor(current_value):
        return value
    level = get_level(current_value)
    base_value = current_value
    batch_dims = []
    batch_sizes = []
    while level >= 1:
        batch_dim = get_batch_dim(base_value)
        base_value, _ = unwrap_batched(base_value, level)
        batch_dims.append(batch_dim)
        batch_sizes.append(base_value.size(batch_dim))
        level -= 1
    batch_dims = tuple(batch_dims[::-1])
    batch_sizes = tuple(batch_sizes[::-1])
    for dim, size in zip(batch_dims, batch_sizes):
        value = value.unsqueeze(dim).expand(*value.shape[:dim], size, *value.shape[dim:])
    expand_value = value
    for level, dim in enumerate(batch_dims, 1):
        value = add_batch_dim(value, dim, level)
    _transform_in_dim(batch_dims, value, expand_value)
    return value


def wrap_vmap_inputs[T: Callable](func: T) -> T:

    @wraps(func)
    def wrapper(*args, **kwargs):
        flat_args, flat_spec = tree_flatten((args, kwargs))
        for arg in flat_args:
            if not isinstance(arg, torch.Tensor):
                continue
            if not is_batched_tensor(arg):
                continue
            level = get_level(arg)
            base_arg = arg
            batch_dims = []
            while level >= 1:
                batch_dim = get_batch_dim(base_arg)
                base_arg, _ = unwrap_batched(base_arg, level)
                batch_dims.append(batch_dim)
                level -= 1
            batch_dims = tuple(batch_dims[::-1])
            _transform_in_dim(batch_dims, arg, base_arg)
        args, kwargs = tree_unflatten(flat_args, flat_spec)
        return func(*args, **kwargs)

    return wrapper


def batched_random(rand_func: Callable, *size: Tuple[int | torch.SymInt], **kwargs) -> torch.Tensor:
    level = current_level()
    if level is None or level <= 0:
        return rand_func(*size, **kwargs)
    # else
    global __vmap_batch_sizes__
    size = tuple(__vmap_batch_sizes__) + size
    num_levels = len(__vmap_batch_sizes__)
    rand_values = rand_func(*size, **kwargs)
    batched_rand_values = rand_values
    for level in range(1, num_levels + 1):
        batched_rand_values = add_batch_dim(batched_rand_values, 0, level)
    _transform_in_dim(tuple(range(num_levels)), batched_rand_values, rand_values)
    return batched_rand_values
