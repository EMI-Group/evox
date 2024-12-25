import torch
from .jit_fix_operator import maximum


def cos_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine distance between two input tensors `x` and `y`.

    Cosine distance is defined as:
        1 - cosine similarity, where cosine similarity is computed as the dot product
        of the normalized vectors.

    Args:
        x (`torch.Tensor`): The first input tensor, typically of shape `(n, d)` where `n` is the number of samples and `d` is the dimensionality.
        y (`torch.Tensor`): The second input tensor, also of shape `(m, d)` where `m` is the number of samples and `d` is the dimensionality.

    Returns:
        `torch.Tensor`: The cosine distance between the input tensors `x` and `y`, with shape `(n, m)`. Each element `(i, j)` represents the cosine distance between `x[i]` and `y[j]`.

    Example:
        ```python
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[4.0, 5.0], [1.0, 2.0]])
        result = cos_dist(x, y)
        print(result)
        ```

    Notes:
        - Cosine distance ranges from 0 (perfectly similar) to 2 (perfectly dissimilar).
        - The function normalizes both `x` and `y` to have unit norm (L2 norm) along the last dimension (`dim=-1`).
    """
    x_normalized = x / torch.norm(x, p=2, dim=-1, keepdim=True)
    y_normalized = y / torch.norm(y, p=2, dim=-1, keepdim=True)
    return torch.matmul(x_normalized, y_normalized.T)


def pairwise_euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_sq = x.pow(2).sum(dim=1, keepdim=True)
    y_sq = y.pow(2).sum(dim=1, keepdim=True)
    dist = x_sq + y_sq.t() - 2.0 * torch.matmul(x, y.t())
    dist = maximum(dist, torch.tensor(0.0, device=dist.device))
    return dist.sqrt()


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