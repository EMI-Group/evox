from typing import Union, List

import torch


def min_by(
    values: List[torch.Tensor],
    keys: List[torch.Tensor],
):
    """
    Find the value with the minimum key.

    Args:
        values: A tensor or list of tensors.
        keys: A tensor or list of tensors corresponding to the values.

    Returns:
        tuple: The value with the minimum key and the corresponding key.
    """
    # if isinstance(values, list):
    values = torch.cat(values, dim=0)
    keys = torch.cat(keys, dim=0)
    min_index = torch.argmin(keys)
    return values[min_index], keys[min_index]

def get_distance_matrix(
    location: torch.Tensor,
):
    """
    Compute the pairwise Euclidean distance matrix.

    Args:
        location: A tensor of shape (N, M), where N is the number of points
                  and M is the number of dimensions.

    Returns:
        A tensor of shape (N, N) containing the pairwise distances.
    """
    assert len(location.shape) == 2

    # Broadcasting without unsqueeze
    diff = location[:, None] - location[None, :]
    dist_matrix = torch.sqrt((diff ** 2).sum(-1))
    return dist_matrix


def row_argsort(
    x: torch.Tensor,
):
    """
    Sort each row of a 2D tensor and return the indices.

    Args:
        x: A tensor of shape (N, M).

    Returns:
        A tensor of shape (N, M) containing the sorted indices for each row.
    """
    assert len(x.shape) == 2

    sorted_indices = torch.argsort(x, dim=-1)
    return sorted_indices


def select_from_mask(mask: torch.Tensor, s: int) -> torch.Tensor:
    """
    Randomly select `s` elements from a 1D mask using uniform noise.

    Args:
        key: A tensor used as the random seed.
        mask: A tensor of shape (N,) containing {0, 1}.
        s: The number of elements to select.

    Returns:
        A new mask tensor with exactly `s` elements set to 1.
    """
    N = mask.shape[0]

    # Generate uniform noise
    noise = torch.rand(N, device=mask.device)

    # Add noise to mask to randomize selection
    sorted_idx = torch.argsort(mask + noise, descending=True)

    # Ensure exactly `s` elements are selected
    result_mask = torch.zeros_like(mask)
    result_mask[sorted_idx[:s]] = 1

    return result_mask
