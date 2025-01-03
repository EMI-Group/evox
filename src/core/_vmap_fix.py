from functools import wraps
from typing import Any, Callable, List, Tuple

import torch
import torch._C._functorch as _functorch
from torch._C._functorch import (
    _add_batch_dim as add_batch_dim,
    maybe_get_bdim as get_batch_dim,
    maybe_get_level as get_level,
    is_batchedtensor as is_batched_tensor,
    _unwrap_batched as unwrap_batched_base,
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
    nn.Buffer = nn.parameter.Buffer


def _set_func_id(new_func, old_func):
    if hasattr(old_func, "__id__"):
        func_id = old_func.__id__
    elif hasattr(old_func, "__self__"):
        func_id = (id(old_func.__self__), id(old_func.__func__))
    else:
        func_id = id(old_func)
    new_func.__id__ = func_id
    

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
        `torch.Tensor`: The batched tensor of random values.

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


def batched_random_like(rand_func: Callable, like_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """Generate a batched tensor of random values with the same shape as the given tensor.

    Given a random function (e.g. [`torch.randn_like`](https://pytorch.org/docs/stable/generated/torch.randn_like.html),
    [`torch.rand_like`](https://pytorch.org/docs/stable/generated/torch.rand_like.html), etc.) and a tensor, this function
    generates a batched tensor of random values by applying the given function to
    the tensor extended with the current vmap batch size.

    Args:
        rand_func (`Callable`): A function that generates a tensor of random values.
        like_tensor (`torch.Tensor`): The tensor to generate random values like.
        **kwargs: The keyword arguments to the given function.

    Returns:
        `torch.Tensor`: The batched tensor of random values.
    """
    level = current_level()
    if level is None or level <= 0:
        return rand_func(like_tensor, **kwargs)
    # else
    original_tensor, batch_dims, _ = unwrap_batch_tensor(like_tensor)
    batch_rand_values = rand_func(original_tensor, **kwargs)
    for level, dim in enumerate(batch_dims):
        batch_rand_values = add_batch_dim(batch_rand_values, dim, level)
    _transform_in_dim(batch_dims, batch_rand_values, original_tensor)
    return batch_rand_values



_original_rand = torch.rand
_original_randn = torch.randn
_original_randint = torch.randint
_original_randperm = torch.randperm
_original_rand_like = torch.rand_like
_original_randn_like = torch.randn_like
_original_randint_like = torch.randint_like
_original_get_item = torch.Tensor.__getitem__
_original_set_item = torch.Tensor.__setitem__


def _batch_rand(*size, **kwargs):
    if "size" in kwargs:
        assert (
            len(size) == 0
        ), f"Expect 0 positional arguments since size is given in kwargs, got {len(size)}"
        size = kwargs.pop("size")
    return batched_random(_original_rand, *size, **kwargs)


def _batch_randn(*size, **kwargs):
    if "size" in kwargs:
        assert (
            len(size) == 0
        ), f"Expect 0 positional arguments since size is given in kwargs, got {len(size)}"
        size = kwargs.pop("size")
    return batched_random(_original_randn, *size, **kwargs)


def _batch_randint(low=None, high=None, size=None, **kwargs):
    assert high is not None and size is not None, "`high` and `size` must be given"
    if low is None:
        low = 0
    return batched_random(_original_randint, *size, low=low, high=high, **kwargs)


def _batch_randperm(n, **kwargs):
    level = current_level()
    if level is None or level <= 0:
        return _original_randperm(n, **kwargs)
    # else
    global __vmap_batch_sizes__
    size = tuple(__vmap_batch_sizes__) + (n,)
    num_levels = len(__vmap_batch_sizes__)
    # rand_values = torch.stack([_original_randperm(n, **kwargs) for _ in torch.arange(prod(size))]).view(size + (n,))
    rand_values = _original_rand(size, **kwargs)
    rand_values = torch.argsort(rand_values, dim=-1)
    rand_values = rand_values.to(**kwargs)
    batched_rand_values = rand_values
    for level in range(1, num_levels + 1):
        batched_rand_values = add_batch_dim(batched_rand_values, 0, level)
    _transform_in_dim(tuple(range(num_levels)), batched_rand_values, rand_values)
    return batched_rand_values


def _batch_rand_like(like_tensor, **kwargs):
    return batched_random_like(_original_rand_like, like_tensor, **kwargs)


def _batch_randn_like(like_tensor, **kwargs):
    return batched_random_like(_original_randn_like, like_tensor, **kwargs)


def _batch_randint_like(like_tensor, **kwargs):
    return batched_random_like(_original_randint_like, like_tensor, **kwargs)


def _batch_randperm(n, **kwargs):
    level = current_level()
    if level is None or level <= 0:
        return _original_randperm(n, **kwargs)
    # else
    global __vmap_batch_sizes__
    size = tuple(__vmap_batch_sizes__) + (n,)
    num_levels = len(__vmap_batch_sizes__)
    rand_values = _original_rand(size, **kwargs)
    rand_values = torch.argsort(rand_values, dim=-1)
    rand_values = rand_values.to(**kwargs)
    batched_rand_values = rand_values
    for level in range(1, num_levels + 1):
        batched_rand_values = add_batch_dim(batched_rand_values, 0, level)
    _transform_in_dim(tuple(range(num_levels)), batched_rand_values, rand_values)
    return batched_rand_values


def _batch_rand_like(like_tensor, **kwargs):
    return batched_random_like(_original_rand_like, like_tensor, **kwargs)


def _batch_randn_like(like_tensor, **kwargs):
    return batched_random_like(_original_randn_like, like_tensor, **kwargs)


def _batch_randint_like(like_tensor, **kwargs):
    return batched_random_like(_original_randint_like, like_tensor, **kwargs)


def _batch_getitem(tensor: torch.Tensor, indices, dim=0):
    if isinstance(indices, torch.Tensor) and indices.ndim <= 1:
        tensor = torch.index_select(tensor, dim, indices)
        if indices.ndim == 0:
            tensor = tensor[*(([slice(None)] * dim) + [0])]
        return tensor
    # default
    return _original_get_item(tensor, indices)


def _batch_setitem(tensor: torch.Tensor, indices, values, dim=0):
    if isinstance(indices, torch.Tensor) and indices.ndim <= 1:
        new_tensor = tensor.scatter(dim, indices, values)
        return tensor.copy_(new_tensor)
    # default
    return _original_set_item(tensor, indices, values)


from contextvars import ContextVar, Token
from contextlib import contextmanager

_batch_fixing: ContextVar[bool] = ContextVar("batch_fixing", default=False)


@contextmanager
def use_batch_fixing(new_batch_fixing: bool = True):
    # Set the new state and obtain a token
    token: Token = _batch_fixing.set(new_batch_fixing)
    torch.rand = _batch_rand if new_batch_fixing else _original_rand
    torch.randn = _batch_randn if new_batch_fixing else _original_randn
    torch.randint = _batch_randint if new_batch_fixing else _original_randint
    torch.randperm = _batch_randperm if new_batch_fixing else _original_randperm
    torch.randperm = _batch_randperm if new_batch_fixing else _original_randperm
    torch.rand_like = _batch_rand_like if new_batch_fixing else _original_rand_like
    torch.randn_like = _batch_randn_like if new_batch_fixing else _original_randn_like
    torch.randint_like = _batch_randint_like if new_batch_fixing else _original_randint_like
    torch.Tensor.__getitem__ = _batch_getitem if new_batch_fixing else _original_get_item
    torch.Tensor.__setitem__ = _batch_setitem if new_batch_fixing else _original_set_item
    try:
        yield token
    finally:
        # Reset the state to its previous value
        _batch_fixing.reset(token)
        torch.rand = _original_rand
        torch.randn = _original_randn
        torch.randint = _original_randint
        torch.randperm = _original_randperm
        torch.rand_like = _original_rand_like
        torch.randn_like = _original_randn_like
        torch.randint_like = _original_randint_like
        torch.Tensor.__getitem__ = _original_get_item
        torch.Tensor.__setitem__ = _original_set_item


def unwrap_batch_tensor(tensor: torch.Tensor):
    """Unwraps a batched tensor into its original tensor and the batch dimensions/sizes.

    Args:
        tensor (`torch.Tensor`): The batched tensor to be unwrapped.

    Returns:
        (`torch.Tensor`, `tuple[int, ...]`, `tuple[int, ...]`): A tuple of the original tensor, the batch dimensions, and the batch sizes.
    """

    level = get_level(tensor)
    batch_dims = []
    batch_sizes = []
    while level >= 1:
        batch_dim = get_batch_dim(tensor)
        tensor, _ = unwrap_batched_base(tensor, level)
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
    def input_args_wrapper(*args, **kwargs):
        flat_args, flat_spec = tree_flatten((args, kwargs))
        for arg in flat_args:
            if not isinstance(arg, torch.Tensor):
                continue
            if not is_batched_tensor(arg):
                continue
            unwrap_arg, batch_dims, _ = unwrap_batch_tensor(arg)
            _transform_in_dim(batch_dims, arg, unwrap_arg)
        args, kwargs = tree_unflatten(flat_args, flat_spec)
        return func(*args, **kwargs)

    _set_func_id(input_args_wrapper, func)
    return input_args_wrapper
