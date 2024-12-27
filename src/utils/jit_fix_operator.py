from typing import List
import torch


def switch(label: torch.Tensor, values: List[torch.Tensor]) -> torch.Tensor:
    """
    Element-wise switch select operator that generates a tensor from a list of tensors based on the label tensor.

    Args:
        label (`torch.Tensor`): A tensor containing labels used to select from the list of tensors. Must be broadcastable to the shape of rest arguments.
        values (`List[torch.Tensor]`): A list of tensors from which one is selected based on the label.
            All tensors in the list must be broadcastable to the same shape.

    Returns:
        `torch.Tensor`: A tensor where each element is selected from the list of tensors based on the
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

    Args:
        a (`torch.Tensor`): The input tensor to be clamped.
        lb (`torch.Tensor`): The lower bound tensor. Must be broadcastable to the shape of `a`.
        ub (`torch.Tensor`): The upper bound tensor. Must be broadcastable to the shape of `a`.

    Returns:
        `torch.Tensor`: A tensor where each element is clamped to be within the specified bounds.
    """
    lb = torch.relu(lb - a)
    ub = torch.relu(a - ub)
    return a + lb - ub


def clip(a: torch.Tensor) -> torch.Tensor:
    """
    Clip the values of the input tensor `a` to be within the range [0, 1].

    Notice: This function invokes `clamp(a, 0, 1)`.

    Args:
        a (`torch.Tensor`): The input tensor to be clipped.

    Returns:
        `torch.Tensor`: A tensor where each element is clipped to be within [0, 1].
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

    Args:
        a (`torch.Tensor`): The first input tensor.
        b (`torch.Tensor`): The second input tensor.

    Returns:
        `torch.Tensor`: The element-wise maximum of `a` and `b`.
    """
    diff = torch.relu(b - a)
    return a + diff


def minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Element-wise minimum of two input tensors `a` and `b`.

    Notice: This is a fix function for [`torch.minimum`](https://pytorch.org/docs/stable/generated/torch.minimum.html] since it is not supported in JIT operator fusion.

    Args:
        a (`torch.Tensor`): The first input tensor.
        b (`torch.Tensor`): The second input tensor.

    Returns:
        `torch.Tensor`: The element-wise minimum of `a` and `b`.
    """
    diff = torch.relu(a - b)
    return a - diff


def lexsort(*keys: torch.Tensor) -> torch.Tensor:
    """
    Perform multi-level lexicographical sorting based on multiple keys.

    This function sorts the given tensors lexicographically, first by the
    first key, then by the second key in case of ties in the first key,
    and so on. It is similar to NumPy's `lexsort` but works with PyTorch
    tensors.

    Args:
        *keys (torch.Tensor): Tensors to be sorted. Each tensor represents
                               a sorting key. All tensors must have the
                               same length.

    Returns:
        torch.Tensor: A tensor of indices that sorts the keys lexicographically.
                      The indices correspond to the order of the sorted
                      elements in the input tensors.

    Raises:
        ValueError: If the input keys have different lengths.

    Example:
        key1 = torch.tensor([1, 3, 2])
        key2 = torch.tensor([9, 7, 8])
        sorted_indices = lexsort(key1, key2)
        # sorted_indices will be the indices that sort the keys first by key2,
        # then by key1 in case of ties.
    """
    key_lengths = [len(key) for key in keys]
    if len(set(key_lengths)) > 1:
        raise ValueError("All input keys must have the same length.")

    sorted_indices = torch.argsort(keys[0])
    for key in (keys[1:]):
        sorted_key = key[sorted_indices]
        final_sorted_indices = torch.argsort(sorted_key)
        sorted_indices = sorted_indices[final_sorted_indices]

    return sorted_indices


def nanmin(input_tensor: torch.Tensor, dim: int = -1, keepdim: bool = False):
    """
    Compute the minimum of a tensor along a specified dimension, ignoring NaN values.

    This function replaces `NaN` values in the input tensor with `infinity` (`float('inf')`),
    and then computes the minimum over the specified dimension, effectively ignoring `NaN` values.

    Args:
        input_tensor (`torch.Tensor`): The input tensor, which may contain `NaN` values.
            It can be of any shape.
        dim (int, optional): The dimension along which to compute the minimum. Default is `-1`,
            which corresponds to the last dimension.
        keepdim (bool, optional): Whether to retain the reduced dimension in the result.
            Default is `False`. If `True`, the output tensor will have the same number of dimensions
            as the input, with the size of the reduced dimension set to 1.

    Returns:
        `torch.return_types.min`: A named tuple with two fields:
            - `values` (`torch.Tensor`): A tensor containing the minimum values computed along the specified dimension,
              ignoring `NaN` values.
            - `indices` (`torch.Tensor`): A tensor containing the indices of the minimum values along the specified dimension.

        The returned tensors `values` and `indices` will have the same shape as the input tensor, except for the dimension(s) over which the operation was performed.

    Example:
        ```python
        x = torch.tensor([[1.0, 2.0], [float('nan'), 4.0]])
        result = nanmin(x, dim=0)
        print(result.values)  # Output: tensor([1.0, 2.0])
        print(result.indices)  # Output: tensor([0, 0])
        ```

    Notes:
        - `NaN` values are ignored by replacing them with `infinity` before computing the minimum.
        - If all values along a dimension are `NaN`, the result will be `infinity` for that dimension, and the index will be returned as the first valid index.
    """
    mask = torch.isnan(input_tensor)
    input_tensor = torch.where(mask, torch.tensor(float('inf'), device=input_tensor.device), input_tensor)
    return input_tensor.min(dim=dim, keepdim=keepdim)


def nanmax(input_tensor: torch.Tensor, dim: int = -1, keepdim: bool = False):
    """
    Compute the maximum of a tensor along a specified dimension, ignoring NaN values.

    This function replaces `NaN` values in the input tensor with `-infinity` (`float('-inf')`),
    and then computes the maximum over the specified dimension, effectively ignoring `NaN` values.

    Args:
        input_tensor (`torch.Tensor`): The input tensor, which may contain `NaN` values.
            It can be of any shape.
        dim (int, optional): The dimension along which to compute the maximum. Default is `-1`,
            which corresponds to the last dimension.
        keepdim (bool, optional): Whether to retain the reduced dimension in the result.
            Default is `False`. If `True`, the output tensor will have the same number of dimensions
            as the input, with the size of the reduced dimension set to 1.

    Returns:
        `torch.return_types.max`: A named tuple with two fields:
            - `values` (`torch.Tensor`): A tensor containing the maximum values computed along the specified dimension,
              ignoring `NaN` values.
            - `indices` (`torch.Tensor`): A tensor containing the indices of the maximum values along the specified dimension.

        The returned tensors `values` and `indices` will have the same shape as the input tensor, except for the dimension(s) over which the operation was performed.

    Example:
        ```python
        x = torch.tensor([[1.0, 2.0], [float('nan'), 4.0]])
        result = nanmax(x, dim=0)
        print(result.values)  # Output: tensor([1.0, 4.0])
        print(result.indices)  # Output: tensor([0, 1])
        ```

    Notes:
        - `NaN` values are ignored by replacing them with `-infinity` before computing the maximum.
        - If all values along a dimension are `NaN`, the result will be `-infinity` for that dimension, and the index will be returned as the first valid index.
    """
    mask = torch.isnan(input_tensor)
    input_tensor = torch.where(mask, torch.tensor(float('-inf'), device=input_tensor.device), input_tensor)
    return input_tensor.max(dim=dim, keepdim=keepdim)