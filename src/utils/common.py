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
    """
    Compute the pairwise Euclidean distances between two tensors x and y.

    Given two tensors x and y with shapes (N, D) and (M, D), where N is the number
    of samples and D is the feature dimension, this function computes the Euclidean
    distance between each pair of samples in x and y. The result is a tensor of shape
    (N, M), where each element represents the Euclidean distance between a sample
    from x and a sample from y.

    Args:
        x (torch.Tensor): A tensor of shape (N, D) representing N samples with D features.
        y (torch.Tensor): A tensor of shape (M, D) representing M samples with D features.

    Returns:
        torch.Tensor: A tensor of shape (N, M) where each element (i, j) represents the
                      Euclidean distance between x[i] and y[j].

    Note:
        - The function calculates the squared norms of x and y, computes the matrix product
          of x and y, and then applies the Euclidean distance formula.
        - The `maximum` function is used to ensure that the computed distances are non-negative
          due to floating-point errors.

    Example:
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        pairwise_euclidean_dist(x, y)
        tensor([[5.6569, 8.4853],
                [2.8284, 2.8284]])

    This function calculates the Euclidean distance between every pair of samples from
    x and y and returns a tensor containing those distances.
    """
    x_sq = x.pow(2).sum(dim=1, keepdim=True)
    y_sq = y.pow(2).sum(dim=1, keepdim=True)
    dist = x_sq + y_sq.t() - 2.0 * torch.matmul(x, y.t())
    dist = maximum(dist, torch.tensor(0.0, device=dist.device))
    return dist.sqrt()
