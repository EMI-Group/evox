from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Callable, List, Sequence, Tuple

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
from torch.utils._pytree import tree_flatten, tree_unflatten  # noqa: F401

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


def _batch_getitem(tensor: torch.Tensor, indices, dim=0):
    level = current_level()
    if level is None or level <= 0:
        return _original_get_item(tensor, indices)
    # else
    if isinstance(indices, torch.Tensor) and indices.ndim <= 1:
        tensor = torch.index_select(tensor, dim, indices)
        if indices.ndim == 0:
            tensor = tensor.__getitem__(*(([slice(None)] * dim) + [0]))
        return tensor
    # default
    return _original_get_item(tensor, indices)


def _batch_setitem(tensor: torch.Tensor, indices, values, dim=0):
    if isinstance(indices, torch.Tensor) and indices.ndim <= 1:
        new_tensor = tensor.scatter(dim, indices, values)
        return tensor.copy_(new_tensor)
    # default
    return _original_set_item(tensor, indices, values)


_batch_fixing: ContextVar[bool] = ContextVar("batch_fixing", default=False)


@contextmanager
def use_batch_fixing(new_batch_fixing: bool = True):
    # Set the new state and obtain a token
    token: Token = _batch_fixing.set(new_batch_fixing)
    torch.Tensor.size = _batch_size if new_batch_fixing else _original_size
    torch.rand = _batch_rand if new_batch_fixing else _original_rand
    torch.randn = _batch_randn if new_batch_fixing else _original_randn
    torch.randint = _batch_randint if new_batch_fixing else _original_randint
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
        torch.Tensor.size = _original_size
        torch.rand = _original_rand
        torch.randn = _original_randn
        torch.randint = _original_randint
        torch.randperm = _original_randperm
        torch.rand_like = _original_rand_like
        torch.randn_like = _original_randn_like
        torch.randint_like = _original_randint_like
        torch.Tensor.__getitem__ = _original_get_item
        torch.Tensor.__setitem__ = _original_set_item


def align_vmap_tensor(value: Any, current_value: Any | None) -> torch.Tensor:
    """
    Aligns a tensor with the batching dimensions of a current batched tensor.

    This function adjusts the input tensor `value` to match the batch dimensions
    of `current_value`, which is assumed to be a batched tensor. If `value` is
    already a batched tensor or `current_value` is not a batched tensor, it
    returns `value` unchanged.

    :param value: The tensor to be aligned. If not a `torch.Tensor`, it is
                    returned unchanged.
    :param current_value: The reference batched tensor. If `None` or
                                not a batched tensor, `value` is returned
                                unchanged.

    :return: The input `value` aligned with the batch dimensions of
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
    value = wrap_batch_tensor(value, batch_dims)
    return value


def _debug_print(format: str, arg: torch.Tensor) -> torch.Tensor:
    print(format.format(arg))
    return arg


def debug_print(format: str, arg: torch.Tensor) -> torch.Tensor:
    """Prints a formatted string with one positional tensor used for debugging inside JIT traced functions on-the-fly.

    When vectorized-mapping, it unwraps the batched tensor to print the underlying values. Otherwise, the function behaves like `format.format(*args, **kwargs)`.

    :param format: A string format.
    :param arg: The positional tensor.
    :return: The unchanged tensor.
    """
    level = current_level()
    if level is None or level <= 0:
        inner_arg = arg
    else:
        inner_arg = unwrap_batch_tensor(arg)[0]
    return torch.jit.script_if_tracing(_debug_print)(format, inner_arg)


debug_print.__prepare_scriptable__ = lambda: _debug_print
