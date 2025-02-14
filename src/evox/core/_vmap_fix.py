__all__ = [
    "add_batch_dim",
    "get_level",
    "current_level",
    "unwrap_batch_tensor",
    "wrap_batch_tensor",
    "register_fix_function",
    "unregister_fix_function",
    "use_batch_fixing",
    "tree_flatten",
    "tree_unflatten",
    "_set_func_id",
]

import math
import warnings
from contextlib import contextmanager
from contextvars import ContextVar, Token
from threading import Lock
from typing import Any, Callable, Dict, List, Sequence, Tuple

# cSpell:words bdim batchedtensor
import torch
import torch._C._functorch as _functorch
import torch._functorch.vmap as vmap
from torch._C._functorch import (
    _add_batch_dim as add_batch_dim,
)
from torch._C._functorch import (
    _unwrap_batched as unwrap_batched_base,
)
from torch._C._functorch import _vmap_decrement_nesting, _vmap_increment_nesting
from torch._C._functorch import (
    is_batchedtensor as is_batched_tensor,
)
from torch._C._functorch import (
    maybe_get_bdim as get_batch_dim,
)
from torch._C._functorch import (
    maybe_get_level as get_level,
)

if "maybe_current_level" not in _functorch.__dict__:

    def current_level() -> int | None:
        try:
            return _functorch.current_level()
        except Exception:
            return None

else:
    current_level = _functorch.maybe_current_level

from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

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


def unwrap_batch_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], Tuple[int, ...]]:
    """Unwraps a batched tensor into its original tensor and the batch dimensions/sizes.

    :param tensor: The batched tensor to be unwrapped.

    :return: A tuple of the original tensor, the batch dimensions, and the batch sizes.
    """
    level = get_level(tensor)
    if level is None or level <= 0:
        return tensor, (), ()
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


def wrap_batch_tensor(tensor: torch.Tensor, in_dims: int | Tuple[int, ...]) -> torch.Tensor:
    """Wraps a original tensor into its batched form with given batch dimensions.

    :param tensor: The original tensor to be wrapped.
    :param in_dims: The batch dimension(s).

    :return: The batched tensor.
    """
    assert get_level(tensor) <= 0, f"Expect vmap level of tensor to be none, got {get_level(tensor)}"
    if not isinstance(in_dims, Sequence):
        in_dims = tuple(in_dims)
    for level, dim in enumerate(in_dims, 1):
        tensor = add_batch_dim(tensor, dim, level)
    return tensor


def _get_batched_size(in_dim: int | Tuple[int, ...], original: torch.Tensor) -> int | Tuple[int, ...] | None:
    if in_dim is None:
        return original.size()
    if not isinstance(in_dim, tuple):
        in_dim = (in_dim,)
    shape = original.size()
    for d in in_dim:
        shape = tuple(s for i, s in enumerate(shape) if i != d)
    return shape


_vmap_batch_sizes: ContextVar[List[int]] = ContextVar("vmap_batch_sizes", default=[])


def get_vmap_batch_sizes():
    return _vmap_batch_sizes.get()


@contextmanager
def vmap_increment_nesting(batch_size, randomness):
    try:
        _vmap_batch_sizes.set(_vmap_batch_sizes.get() + [batch_size])
        vmap_level = _vmap_increment_nesting(batch_size, randomness)
        yield vmap_level
    finally:
        _vmap_decrement_nesting()
        _vmap_batch_sizes.set(_vmap_batch_sizes.get()[:-1])


def _flat_vmap(func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs):
    with vmap_increment_nesting(batch_size, randomness) as vmap_level:
        batched_inputs = vmap._create_batched_inputs(flat_in_dims, flat_args, vmap_level, args_spec)
        batched_outputs = func(*batched_inputs, **kwargs)
        return vmap._unwrap_batched(batched_outputs, out_dims, vmap_level, batch_size, func)


vmap._flat_vmap = _flat_vmap


def batched_random(rand_func: Callable, *size: Tuple[int | torch.SymInt], **kwargs) -> torch.Tensor:
    """
    Generate a batched tensor of random values.

    Given a random function (e.g. [`torch.randn`](https://pytorch.org/docs/stable/generated/torch.randn.html),
    [`torch.rand`](https://pytorch.org/docs/stable/generated/torch.rand.html), etc.) and its size arguments, this function
    generates a batched tensor of random values by applying the given function to
    the size extended with the current vmap batch size.

    :param rand_func: A function that generates a tensor of random values.
    :param *size: The size arguments to the given function.
    :param **kwargs: The keyword arguments to the given function.

    :return: The batched tensor of random values.

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
    size = tuple(get_vmap_batch_sizes()) + size
    num_levels = len(get_vmap_batch_sizes())
    rand_values = rand_func(size=size, **kwargs)
    batched_rand_values = wrap_batch_tensor(rand_values, [0] * num_levels)
    return batched_rand_values


def batched_random_like(rand_func: Callable, like_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """Generate a batched tensor of random values with the same shape as the given tensor.

    Given a random function (e.g. [`torch.randn_like`](https://pytorch.org/docs/stable/generated/torch.randn_like.html),
    [`torch.rand_like`](https://pytorch.org/docs/stable/generated/torch.rand_like.html), etc.) and a tensor, this function
    generates a batched tensor of random values by applying the given function to
    the tensor extended with the current vmap batch size.

    :param rand_func: A function that generates a tensor of random values.
    :param like_tensor: The tensor to generate random values like.
    :param **kwargs: The keyword arguments to the given function.

    :return: The batched tensor of random values.
    """
    level = current_level()
    if level is None or level <= 0:
        return rand_func(like_tensor, **kwargs)
    # else
    original_tensor, batch_dims, _ = unwrap_batch_tensor(like_tensor)
    batched_rand_values = rand_func(original_tensor, **kwargs)
    batched_rand_values = wrap_batch_tensor(batched_rand_values, batch_dims)
    return batched_rand_values


_original_size = torch.Tensor.size
_original_rand = torch.rand
_original_randn = torch.randn
_original_randint = torch.randint
_original_randperm = torch.randperm
_original_rand_like = torch.rand_like
_original_randn_like = torch.randn_like
_original_randint_like = torch.randint_like
_original_get_item = torch.Tensor.__getitem__
_original_set_item = torch.Tensor.__setitem__
_original_reshape = torch.reshape
_original_view = torch.Tensor.view
_original_flatten = torch.flatten
_original_unflatten = torch.unflatten


def _batch_size(tensor: torch.Tensor, dim: int | None = None):
    try:
        return _original_size(tensor, dim)
    except Exception:
        original_tensor, batch_dims, _ = unwrap_batch_tensor(tensor)
        size = _get_batched_size(batch_dims, original_tensor)
        return size if dim is None else size[dim]


def _batch_rand(*size, **kwargs):
    if len(size) == 1 and isinstance(size[0], Sequence):
        size = size[0]
    if "size" in kwargs:
        assert len(size) == 0, f"Expect 0 positional arguments since size is given in kwargs, got {len(size)}"
        size = kwargs.pop("size")
    return batched_random(_original_rand, *size, **kwargs)


def _batch_randn(*size, **kwargs):
    if len(size) == 1 and isinstance(size[0], Sequence):
        size = size[0]
    if "size" in kwargs:
        assert len(size) == 0, f"Expect 0 positional arguments since size is given in kwargs, got {len(size)}"
        size = kwargs.pop("size")
    return batched_random(_original_randn, *size, **kwargs)


def _batch_randint(low=None, high=None, size=None, **kwargs):
    assert high is not None and size is not None, "`high` and `size` must be given"
    if low is None:
        low = 0
    if (isinstance(high, torch.Tensor) and is_batched_tensor(high)) or (
        isinstance(low, torch.Tensor) and is_batched_tensor(low)
    ):
        range: torch.Tensor = high - low
        random_values = batched_random(
            _original_randint, *size, low=torch.iinfo(range.dtype).min, high=torch.iinfo(range.dtype).max, **kwargs
        )
        random_values = random_values % range + low
        return random_values
    else:
        return batched_random(_original_randint, *size, low=low, high=high, **kwargs)


def _batch_randperm(n, **kwargs):
    level = current_level()
    if level is None or level <= 0:
        return _original_randperm(n, **kwargs)
    # else
    size = tuple(get_vmap_batch_sizes()) + (n,)
    num_levels = len(get_vmap_batch_sizes())
    # rand_values = torch.stack([_original_randperm(n, **kwargs) for _ in torch.arange(prod(size))]).view(size + (n,))
    rand_values = _original_rand(size, **kwargs)
    rand_values = torch.argsort(rand_values, dim=-1)
    rand_values = rand_values.to(**kwargs)
    batched_rand_values = rand_values
    batched_rand_values = wrap_batch_tensor(rand_values, [0] * num_levels)
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
    size = tuple(get_vmap_batch_sizes()) + (n,)
    num_levels = len(get_vmap_batch_sizes())
    rand_values = _original_rand(size, **kwargs)
    rand_values = torch.argsort(rand_values, dim=-1)
    rand_values = rand_values.to(**kwargs)
    batched_rand_values = rand_values
    batched_rand_values = wrap_batch_tensor(rand_values, [0] * num_levels)
    return batched_rand_values


def _batch_rand_like(like_tensor, **kwargs):
    return batched_random_like(_original_rand_like, like_tensor, **kwargs)


def _batch_randn_like(like_tensor, **kwargs):
    return batched_random_like(_original_randn_like, like_tensor, **kwargs)


def _batch_randint_like(like_tensor, **kwargs):
    return batched_random_like(_original_randint_like, like_tensor, **kwargs)


def _batch_getitem(tensor: torch.Tensor, indices):
    level = current_level()
    if level is None or level <= 0:
        return _original_get_item(tensor, indices)
    # special case for scalar
    if indices is None and tensor.ndim == 0:
        return tensor.unsqueeze(0)
    # special case for single index
    if isinstance(indices, torch.Tensor):
        tensor = torch.index_select(tensor, 0, indices.flatten())
        if indices.ndim == 0:
            return _original_get_item(tensor, 0)
        else:
            return tensor.unflatten(0, indices.size())
    if not isinstance(indices, Sequence):
        return _original_get_item(tensor, indices)
    # else
    if all(map(lambda ind: isinstance(ind, torch.Tensor) and not ind.dtype.is_floating_point, indices)):
        # https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
        indices: List[torch.Tensor] = list(indices)
        assert all(map(lambda ind: ind.size() == indices[0].size(), indices[1:])), "Expect all index tensors have same shape."
        if get_level(tensor) <= 0:
            return _original_get_item(tensor, indices)
        original_indices = [unwrap_batch_tensor(ind)[0] for ind in indices]
        original_tensor, dims, sizes = unwrap_batch_tensor(tensor)
        for d, s in zip(dims, sizes):
            identity = torch.arange(0, s, dtype=indices[0].dtype, device=tensor.device)
            original_indices.insert(d, (identity,))
        for i, identity in enumerate(original_indices):
            if not isinstance(identity, tuple):
                continue
            original_shape = [1] * original_tensor.ndim
            original_shape[i] = -1
            identity = identity[0].view(*original_shape)
            original_indices[i] = identity
        original_tensor = torch._unsafe_index(original_tensor, original_indices)
        return wrap_batch_tensor(original_tensor, dims)
    # default
    return _original_get_item(tensor, indices)


def _batch_setitem(tensor: torch.Tensor, indices, values, dim=0):
    if isinstance(indices, torch.Tensor) and indices.ndim <= 1:
        new_tensor = tensor.scatter(dim, indices, values)
        return tensor.copy_(new_tensor)
    # default
    return _original_set_item(tensor, indices, values)


def _get_original_dims(tensor: torch.Tensor, batch_dims: Tuple[int], batch_sizes: Tuple[int]):
    ori_shape = list(tensor.size())
    for i, (d, s) in enumerate(zip(batch_dims, batch_sizes)):
        ori_shape.insert(d, (s, i))
    ori_dims = [-1] * len(batch_dims)
    for i, s in enumerate(ori_shape):
        if isinstance(s, tuple):
            ori_shape[i] = s[0]
            ori_dims[s[1]] = i
    return tuple(ori_dims)


def _special_reshape(ori_tensor: torch.Tensor, ori_dims: Tuple[int], new_shape: Tuple[int]):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # check
        original_ndim = ori_tensor.ndim
        for d in ori_dims:
            assert 0 <= d < original_ndim
        assert len(set(ori_dims)) == len(ori_dims)
        # get remaining dims
        preserved_dims = list(ori_dims)
        remaining_dims = [d for d in range(original_ndim) if d not in ori_dims]
        permute_order = preserved_dims + remaining_dims
        # get actual new shape
        preserved_shape = [ori_tensor.size(d) for d in preserved_dims]
        remaining_size = 1
        for d in remaining_dims:
            remaining_size *= ori_tensor.size(d)
        # check new shape
        new_shape_prod = 1
        for s in new_shape:
            assert new_shape_prod > 0 or s > 0, "Cannot have multiple dimensions with size=-1"
            new_shape_prod *= s
        assert remaining_size == new_shape_prod or (new_shape_prod < 0 and remaining_size % (-new_shape_prod) == 0), (
            f"Cannot reshape size {remaining_size} to {new_shape} ({ori_dims}, {ori_tensor.size()})"
        )
        # permute and reshape
        permuted = ori_tensor.permute(*permute_order)
        return _original_reshape(permuted, preserved_shape + list(new_shape))


def _batch_reshape(tensor: torch.Tensor, *shape):
    if isinstance(shape[0], Sequence):
        shape = shape[0]
    level = get_level(tensor)
    if level is None or level <= 0:
        return _original_reshape(tensor, shape)
    # else
    original_tensor, dims, sizes = unwrap_batch_tensor(tensor)
    ori_dims = _get_original_dims(tensor, dims, sizes)
    new_tensor = _special_reshape(original_tensor, ori_dims, shape)
    return wrap_batch_tensor(new_tensor, tuple(range(len(ori_dims))))


def _batch_view(tensor: torch.Tensor, *args, **kwargs):
    if "dtype" in kwargs or isinstance(args[0], torch.dtype):
        return _original_view(tensor, *args, **kwargs)
    level = get_level(tensor)
    if level is None or level <= 0:
        return _original_view(tensor, *args, **kwargs)
    # else
    shape = kwargs.get("size", None) or args
    if isinstance(shape, Sequence) and isinstance(shape[0], Sequence):
        shape = shape[0]
    original_tensor, dims, sizes = unwrap_batch_tensor(tensor)
    ori_dims = _get_original_dims(tensor, dims, sizes)
    new_tensor = _special_reshape(original_tensor, ori_dims, shape)
    return wrap_batch_tensor(new_tensor, tuple(range(len(ori_dims))))


def _batch_flatten(tensor: torch.Tensor, start_dim=0, end_dim=-1):
    level = get_level(tensor)
    if level is None or level <= 0:
        return _original_flatten(tensor, start_dim, end_dim)
    # else
    original_tensor, dims, sizes = unwrap_batch_tensor(tensor)
    ori_dims = _get_original_dims(tensor, dims, sizes)
    shape = list(tensor.size())
    if end_dim not in [-1, tensor.ndim - 1]:
        shape = shape[:start_dim] + [math.prod(shape[start_dim:end_dim + 1])] + shape[end_dim + 1:]
    else:
        shape = shape[:start_dim] + [math.prod(shape[start_dim:])]
    new_tensor = _special_reshape(original_tensor, ori_dims, shape)
    return wrap_batch_tensor(new_tensor, tuple(range(len(ori_dims))))


def _batch_unflatten(tensor: torch.Tensor, dim: int, unflattened_size: Sequence[int]):
    level = get_level(tensor)
    if level is None or level <= 0:
        return _original_unflatten(tensor, dim, unflattened_size)
    # else
    original_tensor, dims, sizes = unwrap_batch_tensor(tensor)
    ori_dims = _get_original_dims(tensor, dims, sizes)
    shape = list(tensor.size())
    if dim not in [-1, tensor.ndim - 1]:
        shape = shape[:dim] + list(unflattened_size) + shape[dim + 1 :]
    else:
        shape = shape[:dim] + list(unflattened_size)
    new_tensor = _special_reshape(original_tensor, ori_dims, shape)
    return wrap_batch_tensor(new_tensor, tuple(range(len(ori_dims))))


_fix_functions_lock = Lock()
_fix_functions: Dict[str, Tuple[Any, Callable, Callable]] = {}


def register_fix_function(namespace: str, name: str, fix_function: Callable):
    """Register a function that fixes a PyTorch function for `torch.vmap`.

    :param namespace: The namespace of the function to be fixed, e.g. "torch".
    :param name: The name of the function to be fixed, e.g. "randn_like".
    :param fix_function: The function that fixes the original function.

    :raises AssertionError: If the specified function is not callable.
    """
    namespace_obj = eval(namespace)
    original_function = getattr(namespace_obj, name, None)
    assert original_function is not None and callable(original_function), f"{namespace}.{name} is not callable"
    _fix_functions_lock.acquire()
    try:
        _fix_functions[f"{namespace}.{name}"] = (namespace_obj, original_function, fix_function)
    finally:
        _fix_functions_lock.release()


def unregister_fix_function(namespace: str, name: str):
    """Unregister a function that fixes a PyTorch function for `torch.vmap`.

    :param namespace: The namespace of the function to be unregistered, e.g. "torch".
    :param name: The name of the function to be unregistered, e.g. "randn_like".
    """
    _fix_functions_lock.acquire()
    try:
        del _fix_functions[f"{namespace}.{name}"]
    finally:
        _fix_functions_lock.release()


register_fix_function("torch", "rand", _batch_rand)
register_fix_function("torch", "randn", _batch_randn)
register_fix_function("torch", "randint", _batch_randint)
register_fix_function("torch", "randperm", _batch_randperm)
register_fix_function("torch", "rand_like", _batch_rand_like)
register_fix_function("torch", "randn_like", _batch_randn_like)
register_fix_function("torch", "randint_like", _batch_randint_like)
register_fix_function("torch.Tensor", "__getitem__", _batch_getitem)
register_fix_function("torch.Tensor", "__setitem__", _batch_setitem)
register_fix_function("torch.Tensor", "size", _batch_size)

register_fix_function("torch.Tensor", "reshape", _batch_reshape)
register_fix_function("torch.Tensor", "view", _batch_view)
register_fix_function("torch.Tensor", "flatten", _batch_flatten)
register_fix_function("torch.Tensor", "unflatten", _batch_unflatten)
register_fix_function("torch", "reshape", _batch_reshape)
register_fix_function("torch", "flatten", _batch_flatten)
register_fix_function("torch", "unflatten", _batch_unflatten)


_batch_fixing: ContextVar[bool] = ContextVar("batch_fixing", default=False)


@contextmanager
def use_batch_fixing(new_batch_fixing: bool = True):
    # Set the new state and obtain a token
    token: Token = _batch_fixing.set(new_batch_fixing)
    if new_batch_fixing:
        for name, (namespace_obj, _, fix_function) in _fix_functions.items():
            setattr(namespace_obj, name.split(".")[-1], fix_function)
    try:
        yield token
    finally:
        # Reset the state to its previous value
        _batch_fixing.reset(token)
        for name, (namespace_obj, original_function, _) in _fix_functions.items():
            setattr(namespace_obj, name.split(".")[-1], original_function)


def align_vmap_tensor(value: Any, current_value: Any | None) -> torch.Tensor:
    """
    Aligns a tensor with the batching dimensions of a current batched tensor.

    This function adjusts the input tensor `value` to match the batch dimensions
    of `current_value`, which is assumed to be a batched tensor. If `value` is
    already a batched tensor or `current_value` is not a batched tensor, it
    returns `value` unchanged.

    :param value: The tensor to be aligned. If not a `torch.Tensor`, it is returned unchanged.
    :param current_value: The reference batched tensor. If `None` or not a batched tensor,
                          `value` is returned unchanged.

    :return: The input `value` aligned with the batch dimensions of `current_value`, if applicable.
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
    value = wrap_batch_tensor(value, batch_dims)
    return value
