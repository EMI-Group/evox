from typing import Any, Protocol, Sequence, Tuple, TypeVar

import torch
from torch._functorch.autograd_function import VmapInfo

from .re_export import tree_flatten, tree_unflatten

T = TypeVar("T", bound=torch.Tensor | Any)


class FnCallable(Protocol[T]):
    def __call__(self, *args: T) -> T | Tuple[T, ...]:
        pass


class VmapFnCallable(Protocol[T]):
    def __call__(
        self, info: VmapInfo, in_dims: Tuple[int | None, ...], *args: T
    ) -> Tuple[T, int | None] | Tuple[Tuple[T, int | None], ...]:
        pass


class VmapWrapInputsCallable(Protocol[T]):
    def __call__(self, info: VmapInfo, in_dims: Tuple[int | None, ...], *args: T) -> Tuple[T, ...]:
        pass


def _default_vmap_wrap_inputs(info: VmapInfo, in_dims: Tuple[int | None, ...], *args):
    new_args = []
    for d, x in zip(in_dims, args):
        if d is None:
            if isinstance(x, torch.Tensor):
                x = x.unsqueeze(0)
            else:
                flat_arg, tree_spec = tree_flatten(x)
                x = tree_unflatten(
                    [(a.unsqueeze(0) if isinstance(a, torch.Tensor) else a) for a in flat_arg], tree_spec
                )
        else:
            if isinstance(x, torch.Tensor):
                x = x.movedim(d, 0)
            else:
                flat_arg, tree_spec = tree_flatten(x)
                x = tree_unflatten(
                    [(a.movedim(d, 0) if isinstance(a, torch.Tensor) else a) for a in flat_arg], tree_spec
                )
        new_args.append(x)
    return tuple(new_args)


def register_vmap_op(
    fn: FnCallable | None = None,
    /,
    *,
    fake_fn: FnCallable | None = None,
    vmap_fn: VmapFnCallable | None = None,
    fake_vmap_fn: VmapFnCallable | None = None,
    vmap_wrap_inputs: VmapWrapInputsCallable | None = None,
    name: str = None,
    mutates_args: str | Sequence[str] = (),
    device_types: str | Sequence[str] | None = None,
    schema: str | None = None,
):
    """
    Register a function as a custom operator with (optional) vectorized-map (vmap) support.
    This function is a simple wrapper around [`torch.library.custom_op`](https://pytorch.org/docs/stable/library.html#torch.library.custom_op),
    see [PyTorch Custom Op](https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial) and [`torch.library.custom_op`](https://pytorch.org/docs/stable/library.html#torch.library.custom_op) for more information.

    :param fn: The operator function to register.
    :param fake_fn: The fake (abstract evaluation) function to register to `fn`.
    :param vmap_fn: The vmap function to register to `fn`. Default None means no vmap support.
    :param fake_vmap_fn: The fake (abstract evaluation) vmap function to register to `vmap_fn`. Ignored if `vmap_fn` is None; **cannot** be None otherwise.
    :param vmap_wrap_inputs: The function to deal with inputs for `vmap_fn`. Ignored if `vmap_fn` is None. Default None will be replaced by `_default_vmap_wrap_inputs`, which moves all inputs's vmap dimensions to the first dimensions (including pytree leafs), and adds additional broadcast dimensions at the beginning.
    :param name: The name of the operator. Default None will be replaced by `"evox::_custom_op_" + fn.__name__`.
    :param mutates_args: The names of args that the function mutates. This MUST be accurate, otherwise, the behavior is undefined. See `torch.library.custom_op` for more information.
    :param device_types: The device types that the operator supports. See `torch.library.custom_op` for more information.
    :param schema: The schema of the operator. See `torch.library.custom_op` for more information.

    ## Example
    ```python
    def _fake_eval(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.new_empty(b.size())

    def _fake_vmap_eval(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, int]:
        return _fake_eval(a, b), 0

    def _vmap_eval(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, int]:
        return a * b.sum(dim=1, keepdim=True), 0

    @register_vmap_op(fake_fn=_fake_eval, vmap_fn=_vmap_eval)
    def _custom_eval(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b.sum()
    ```
    """
    kwargs = {"mutates_args": mutates_args, "device_types": device_types, "schema": schema}
    vmap_registered: torch.library.CustomOpDef = None

    def register(fn: FnCallable):
        nonlocal name, fake_fn, vmap_fn, fake_vmap_fn, vmap_wrap_inputs, vmap_registered
        if name is None:
            name = "evox::_custom_op_" + fn.__name__
        registered: torch.library.CustomOpDef = torch.library.custom_op(name, fn, **kwargs)
        assert fake_fn is not None, "`fake_fn` cannot be None"
        registered.register_fake(fake_fn)
        if vmap_fn is None:
            return registered
        # else
        assert fake_vmap_fn is not None, "`fake_vmap_fn` cannot be None"
        vmap_name = name + "_vmap"
        if vmap_wrap_inputs is None:
            vmap_wrap_inputs = _default_vmap_wrap_inputs
        vmap_registered = torch.library.custom_op(vmap_name, vmap_fn, **kwargs)
        vmap_registered.register_fake(fake_vmap_fn)
        registered.register_vmap(
            lambda info, in_dims, *args: vmap_registered(*vmap_wrap_inputs(info, in_dims, *args))
        )
        return registered

    if fn is None:
        return register
    else:
        return register(fn)
