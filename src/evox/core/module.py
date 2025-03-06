from functools import wraps
from typing import Callable, Optional, TypeVar, Union

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode


def _if_none(a, b):
    return b if a is None else a


def _is_magic(name: str):
    return name.startswith("__") and name.endswith("__")


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


@wraps(torch.vmap)
def vmap(*args, **kwargs) -> Callable:
    """Fix the torch.vmap's issue with __getitem__ and __setitem__.
    Related issue: https://github.com/pytorch/pytorch/issues/124423.
    """

    vmapped = torch.vmap(*args, **kwargs)

    def wrapper(*args, **kwargs):
        with TransformGetSetItemToIndex():
            return vmapped(*args, **kwargs)

    return wrapper


def use_state(stateful_func: Union[Callable, nn.Module]) -> Callable:
    """Transform a `torch.nn.Module`'s method or an `torch.nn.Module` into a stateful function.
    When using `torch.nn.Module`, the stateful version of the default `forward` method will be created.
    The stateful function will have a signature of `fn(params_and_buffers, *args, **kwargs) -> params_and_buffers | Tuple[params_and_buffers, <original_returns>]]`.

    :note: This function **in-place** sets the `forward` method of the underlying `torch.nn.Module` to `stateful_func`. Therefore, an additional `reset_forward` method is provided to restore the original `forward` method.

    ## Examples

    ```python
    from evox import use_state, vmap
    workflow = ... # define your workflow
    stateful_step = use_state(workflow.step)
    vmap_stateful_step = vmap(stateful_step)
    batch_state = torch.func.stack_module_states([workflow] * 3)
    new_batch_state = vmap_stateful_step(batch_state)
    # if your workflow has `forward` that you want to invoke, use "reset_forward" to restore it
    stateful_step.reset_forward()
    workflow.forward(...)
    ```
    """
    if not isinstance(stateful_func, torch.nn.Module):
        module: torch.nn.Module = stateful_func.__self__
        assert isinstance(module, torch.nn.Module), (
            "`stateful_func` must be a `torch.nn.Module` or a method of a `torch.nn.Module`"
        )
        ori_forward = module.forward
        module.forward = stateful_func

        def reset_forward():
            return setattr(module, "forward", ori_forward)
    else:
        module = stateful_func

        def reset_forward():
            pass

    def wrapper(params_and_buffers, *args, **kwargs):
        output = torch.func.functional_call(module, params_and_buffers, *args, **kwargs)
        if output is None:
            return params_and_buffers
        else:
            return params_and_buffers, output

    wrapper.reset_forward = reset_forward

    return wrapper
