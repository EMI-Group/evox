from typing import List

import torch


def switch(label: torch.Tensor, values: List[torch.Tensor]) -> torch.Tensor:
    """
    Element-wise switch select operator that generates a tensor from a list of tensors based on the label tensor.

    :param label: A tensor containing labels used to select from the list of tensors. Must be broadcastable to the shape of rest arguments.
    :param values: A list of tensors from which one is selected based on the label.
        All tensors in the list must be broadcastable to the same shape.

    :return: A tensor where each element is selected from the list of tensors based on the
            corresponding element in the label tensor.
    """
    num_labels = len(values)
    value = values[0]
    for i in range(1, num_labels):
        value = torch.where(label <= i - 1, value, values[i])
    return value


def clamp(a: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """
    Clamp the values of the input tensor `a` to be within the given lower (`lb`) and upper (`ub`) bounds.

    This function ensures that each element of the tensor `a` is not less than the corresponding element
    of `lb` and not greater than the corresponding element of `ub`.

    ## Notice:
    1. This is a fix function for [`torch.clamp`](https://pytorch.org/docs/stable/generated/torch.clamp.html) since it is not supported in JIT operator fusion.
    2. This is NOT a precise replication of `torch.clamp` if `a`, `lb` or `ub` is a float tensor and may suffer from numerical precision losses. Please use `torch.clamp` instead if a precise clamp is required.

    :param a: The input tensor to be clamped.
    :param lb: The lower bound tensor. Must be broadcastable to the shape of `a`.
    :param ub: The upper bound tensor. Must be broadcastable to the shape of `a`.

    :return: A tensor where each element is clamped to be within the specified bounds.
    """
    lb = torch.relu(lb - a)
    ub = torch.relu(a - ub)
    return a + lb - ub


def clip(a: torch.Tensor) -> torch.Tensor:
    """
    Clip the values of the input tensor `a` to be within the range [0, 1].

    Notice: This function invokes `clamp(a, 0, 1)`.

    :param a: The input tensor to be clipped.

    :return: A tensor where each element is clipped to be within [0, 1].
    """
    return clamp(
        a,
        torch.zeros((), dtype=a.dtype, device=a.device),
        torch.ones((), dtype=a.dtype, device=a.device),
    )


def maximum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Element-wise maximum of two input tensors `a` and `b`.

    Notice: This is a fix function for [`torch.maximum`](https://pytorch.org/docs/stable/generated/torch.maximum.html] since it is not supported in JIT operator fusion.

    :param a: The first input tensor.
    :param b: The second input tensor.

    :return: The element-wise maximum of `a` and `b`.
    """
    diff = torch.relu(b - a)
    return a + diff


def minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Element-wise minimum of two input tensors `a` and `b`.

    Notice: This is a fix function for [`torch.minimum`](https://pytorch.org/docs/stable/generated/torch.minimum.html] since it is not supported in JIT operator fusion.

    :param a: The first input tensor.
    :param b: The second input tensor.

    :return: The element-wise minimum of `a` and `b`.
    """
    diff = torch.relu(a - b)
    return a - diff
