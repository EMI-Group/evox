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

    ## Notice
    This is a fix function for [`torch.clamp`](https://pytorch.org/docs/stable/generated/torch.clamp.html) since it is not supported in JIT operator fusion yet.
    ## Warning
    This is NOT a precise replication of `torch.clamp` if `a`, `lb` or `ub` is a float tensor and may suffer from numerical precision losses. Please use `torch.clamp` instead if a precise clamp is required.

    :param a: The input tensor to be clamped.
    :param lb: The lower bound tensor. Must be broadcastable to the shape of `a`.
    :param ub: The upper bound tensor. Must be broadcastable to the shape of `a`.

    :return: A tensor where each element is clamped to be within the specified bounds.
    """
    lb = torch.relu(lb - a)
    ub = torch.relu(a - ub)
    return a + lb - ub


def clamp_float(a: torch.Tensor, lb: float, ub: float) -> torch.Tensor:
    """
    Clamp the float values of the input tensor `a` to be within the given lower (`lb`) and upper (`ub`) bounds.

    This function ensures that each element of the tensor `a` is not less than `lb` and not greater than `ub`.

    ## Notice
    This is a fix function for [`torch.clamp`](https://pytorch.org/docs/stable/generated/torch.clamp.html) since it is not supported in JIT operator fusion yet.
    ## Warning
    This is NOT a precise replication of `torch.clamp` if `a` is a float tensor and may suffer from numerical precision losses. Please use `torch.clamp` instead if a precise clamp is required.

    :param a: The input tensor to be clamped.
    :param lb: The lower bound value. Each element of `a` will be clamped to be not less than `lb`.
    :param ub: The upper bound value. Each element of `a` will be clamped to be not greater than `ub`.

    :return: A tensor where each element is clamped to be within the specified bounds.
    """
    lb = torch.relu(lb - a)
    ub = torch.relu(a - ub)
    return a + lb - ub


def clamp_int(a: torch.Tensor, lb: int, ub: int) -> torch.Tensor:
    """
    Clamp the int values of the input tensor `a` to be within the given lower (`lb`) and upper (`ub`) bounds.

    This function ensures that each element of the tensor `a` is not less than `lb` and not greater than `ub`.

    ## Notice
    This is a fix function for [`torch.clamp`](https://pytorch.org/docs/stable/generated/torch.clamp.html) since it is not supported in JIT operator fusion yet.
    ## Warning
    This is NOT a precise replication of `torch.clamp` if `a` is a float tensor and may suffer from numerical precision losses. Please use `torch.clamp` instead if a precise clamp is required.

    :param a: The input tensor to be clamped.
    :param lb: The lower bound value. Each element of `a` will be clamped to be not less than `lb`.
    :param ub: The upper bound value. Each element of `a` will be clamped to be not greater than `ub`.

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

    ## Notice
    This is a fix function for [`torch.maximum`](https://pytorch.org/docs/stable/generated/torch.maximum.html] since it is not supported in JIT operator fusion yet.
    ## Warning
    This is NOT a precise replication of `torch.maximum` if `a` or `b` is a float tensor and may suffer from numerical precision losses. Please use `torch.maximum` instead if a precise maximum is required.

    :param a: The first input tensor.
    :param b: The second input tensor.

    :return: The element-wise maximum of `a` and `b`.
    """
    diff = torch.relu(b - a)
    return a + diff


def minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Element-wise minimum of two input tensors `a` and `b`.

    ## Notice
    This is a fix function for [`torch.minimum`](https://pytorch.org/docs/stable/generated/torch.minimum.html] since it is not supported in JIT operator fusion yet.
    ## Warning
    This is NOT a precise replication of `torch.minimum` if `a` or `b` is a float tensor and may suffer from numerical precision losses. Please use `torch.minimum` instead if a precise minimum is required.

    :param a: The first input tensor.
    :param b: The second input tensor.

    :return: The element-wise minimum of `a` and `b`.
    """
    diff = torch.relu(a - b)
    return a - diff


def maximum_float(a: torch.Tensor, b: float) -> torch.Tensor:
    """
    Element-wise maximum of input tensor `a` and float `b`.

    ## Notice
    This is a fix function for [`torch.maximum`](https://pytorch.org/docs/stable/generated/torch.maximum.html] since it is not supported in JIT operator fusion yet.
    ## Warning
    This is NOT a precise replication of `torch.maximum` if `a` is a float tensor and may suffer from numerical precision losses. Please use `torch.maximum` instead if a precise maximum is required.

    :param a: The first input tensor.
    :param b: The second input float, which is a scalar value.

    :return: The element-wise maximum of `a` and `b`.
    """
    diff = torch.relu(b - a)
    return a + diff


def minimum_float(a: torch.Tensor, b: float) -> torch.Tensor:
    """
    Element-wise minimum of input tensor `a` and float `b`.

    ## Notice
    This is a fix function for [`torch.minimum`](https://pytorch.org/docs/stable/generated/torch.minimum.html] since it is not supported in JIT operator fusion yet.
    ## Warning
    This is NOT a precise replication of `torch.minimum` if `a` is a float tensor and may suffer from numerical precision losses. Please use `torch.minimum` instead if a precise minimum is required.

    :param a: The first input tensor.
    :param b: The second input float, which is a scalar value.

    :return: The element-wise minimum of `a` and `b`.
    """
    diff = torch.relu(a - b)
    return a - diff


def maximum_int(a: torch.Tensor, b: int) -> torch.Tensor:
    """
    Element-wise maximum of input tensor `a` and int `b`.

    ## Notice
    This is a fix function for [`torch.maximum`](https://pytorch.org/docs/stable/generated/torch.maximum.html] since it is not supported in JIT operator fusion yet.
    ## Warning
    This is NOT a precise replication of `torch.maximum` if `a` is a float tensor and may suffer from numerical precision losses. Please use `torch.maximum` instead if a precise maximum is required.

    :param a: The first input tensor.
    :param b: The second input int, which is a scalar value.

    :return: The element-wise maximum of `a` and `b`.
    """
    diff = torch.relu(b - a)
    return a + diff


def minimum_int(a: torch.Tensor, b: int) -> torch.Tensor:
    """
    Element-wise minimum of input tensor `a` and int `b`.

    ## Notice
    This is a fix function for [`torch.minimum`](https://pytorch.org/docs/stable/generated/torch.minimum.html] since it is not supported in JIT operator fusion yet.
    ## Warning
    This is NOT a precise replication of `torch.minimum` if `a` is a float tensor and may suffer from numerical precision losses. Please use `torch.minimum` instead if a precise minimum is required.

    :param a: The first input tensor.
    :param b: The second input int, which is a scalar value.

    :return: The element-wise minimum of `a` and `b`.
    """
    diff = torch.relu(a - b)
    return a - diff


def lexsort(keys: List[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    Perform lexicographical sorting of multiple tensors, considering each tensor as a key.

    This function sorts the given tensors lexicographically, where sorting is performed
    by the first key, then by the second key in case of ties in the first key, and so on.
    It works similarly to NumPy's `lexsort`, but is designed for PyTorch tensors.

    :param keys: A list of tensors to be sorted, where each tensor represents
                                a sorting key. All tensors must have the same length along
                                the specified dimension (`dim`).
    :param dim: The dimension along which to perform the sorting. Defaults to -1 (the last dimension).

    :return: A tensor containing indices that will sort the input tensors lexicographically.
                      These indices indicate the order of elements in the sorted tensors.

    ## Example
    ```
    key1 = torch.tensor([1, 3, 2])
    key2 = torch.tensor([9, 7, 8])
    sorted_indices = lexsort([key1, key2])
    # sorted_indices will contain the indices that sort first by key2,
    # and then by key1 in case of ties.
    ```

    :note: You can use `torch.unbind` to split the tensor into list.
    """

    sorted_indices = torch.argsort(keys[0], dim=dim, stable=True)
    for key in keys[1:]:
        sorted_key = key.gather(dim, sorted_indices)
        final_sorted_indices = torch.argsort(sorted_key, dim=dim, stable=True)
        sorted_indices = sorted_indices.gather(dim, final_sorted_indices)

    return sorted_indices


def nanmin(input_tensor: torch.Tensor, dim: int = -1, keepdim: bool = False):
    """
    Compute the minimum of a tensor along a specified dimension, ignoring NaN values.

    This function replaces `NaN` values in the input tensor with `infinity` ,
    and then computes the minimum over the specified dimension, effectively ignoring `NaN` values.

    :param input_tensor: The input tensor, which may contain `NaN` values.
        It can be of any shape.
    :param dim: The dimension along which to compute the minimum. Default is `-1`,
        which corresponds to the last dimension.
    :param keepdim: Whether to retain the reduced dimension in the result.
        Default is `False`. If `True`, the output tensor will have the same number of dimensions
        as the input, with the size of the reduced dimension set to 1.

    :return: A named tuple with two fields:
        - `values` (`torch.Tensor`): A tensor containing the minimum values computed along the specified dimension,
            ignoring `NaN` values.
        - `indices` (`torch.Tensor`): A tensor containing the indices of the minimum values along the specified dimension.

        The returned tensors `values` and `indices` will have the same shape as the input tensor, except for the dimension(s) over which the operation was performed.

    ## Example
    ```python
    x = torch.tensor([[1.0, 2.0], [float('nan'), 4.0]])
    result = nanmin(x, dim=0)
    print(result.values)  # Output: tensor([1.0, 2.0])
    print(result.indices)  # Output: tensor([0, 0])
    ```

    ```{note}
    - `NaN` values are ignored by replacing them with `infinity` before computing the minimum.
    - If all values along a dimension are `NaN`, the result will be `infinity` for that dimension, and the index will be returned as the first valid index.
    ```
    """
    mask = torch.isnan(input_tensor)
    input_tensor = torch.where(mask, torch.inf, input_tensor)
    return input_tensor.min(dim=dim, keepdim=keepdim)


def nanmax(input_tensor: torch.Tensor, dim: int = -1, keepdim: bool = False):
    """
    Compute the maximum of a tensor along a specified dimension, ignoring NaN values.

    This function replaces `NaN` values in the input tensor with `-infinity`,
    and then computes the maximum over the specified dimension, effectively ignoring `NaN` values.

    :param input_tensor: The input tensor, which may contain `NaN` values.
        It can be of any shape.
    :param dim: The dimension along which to compute the maximum. Default is `-1`,
        which corresponds to the last dimension.
    :param keepdim: Whether to retain the reduced dimension in the result.
        Default is `False`. If `True`, the output tensor will have the same number of dimensions
        as the input, with the size of the reduced dimension set to 1.

    :return: A named tuple with two fields:
        - `values` (`torch.Tensor`): A tensor containing the maximum values computed along the specified dimension,
              ignoring `NaN` values.
        - `indices` (`torch.Tensor`): A tensor containing the indices of the maximum values along the specified dimension.

        The returned tensors `values` and `indices` will have the same shape as the input tensor, except for the dimension(s) over which the operation was performed.

    ## Example
    ```python
    x = torch.tensor([[1.0, 2.0], [float('nan'), 4.0]])
    result = nanmax(x, dim=0)
    print(result.values)  # Output: tensor([1.0, 4.0])
    print(result.indices)  # Output: tensor([0, 1])
    ```

    ```{note}
    - `NaN` values are ignored by replacing them with `-infinity` before computing the maximum.
    - If all values along a dimension are `NaN`, the result will be `-infinity` for that dimension, and the index will be returned as the first valid index.
    ```
    """
    mask = torch.isnan(input_tensor)
    input_tensor = torch.where(mask, -torch.inf, input_tensor)
    return input_tensor.max(dim=dim, keepdim=keepdim)
