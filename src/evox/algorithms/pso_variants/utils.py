from typing import List

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
    values = torch.cat(values, dim=0)
    keys = torch.cat(keys, dim=0)
    min_index = torch.argmin(keys)
    return values[min_index], keys[min_index]


def random_select_from_mask(mask: torch.Tensor, count: int, dim: int=-1) -> torch.Tensor:
    """
    Randomly select `s` elements from a 1D mask using uniform noise.

    Args:
        key: A tensor used as the random seed.
        mask: A tensor of shape (N,) containing {0, 1}.
        s: The number of elements to select.

    Returns:
        A new mask tensor with exactly `s` elements set to 1.
    """
    assert mask.dtype == torch.bool, f"Expected mask to be boolean, got {mask.dtype}"
    # Add noise to mask to randomize selection
    noise = torch.rand_like(mask)
    sorted_idx = torch.argsort(mask + noise, dim=dim, descending=True)
    sorted_idx = torch.slice_copy(sorted_idx, dim, end=count)
    result_mask = torch.zeros_like(mask)
    result_mask = result_mask.scatter(dim, sorted_idx, 1)
    return result_mask
