"""Functional ETL port of ``src/evox/algorithms/so/pso_variants/utils.py``.

Plain (non-defn) functions — callable only inside an active etl trace.
Semantics mirror the torch originals 1:1 (see DESIGN.md §4.3).
"""

from typing import List, Tuple

import etl
import etl.numpy as enp
import etl.random as random

Tensor = etl.SymbolicTensor


def min_by(values: List[Tensor], keys: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """Find the value with the minimum key.

    Concatenates along axis 0, takes the argmin over the concatenated keys and
    returns the corresponding value (axis-0 row squeezed away) and the minimum
    key itself — the torch ``min_by`` semantics (cat + argmin + index_select).
    """
    values = etl.concatenate(values, axis=0)
    keys = etl.concatenate(keys, axis=0)
    idx = etl.argmin(keys, axis=0)
    idx = enp.reshape(idx, (1,))
    return (
        enp.reshape(etl.gather(values, idx, axis=0), tuple(values.shape[1:])),
        enp.reshape(etl.gather(keys, idx, axis=0), ()),
    )


def random_select_from_mask(
    key: Tensor, mask: Tensor, count: int, dim: int = -1
) -> Tensor:
    """Randomly select ``count`` True entries from a boolean mask.

    Adds uniform noise to the mask, sorts descending by (mask + noise), keeps
    the first ``count`` indices and scatters 1 into a zero mask there — the
    torch ``random_select_from_mask`` semantics with key-first RNG.
    """
    assert mask.dtype == etl.bool_, f"Expected mask to be boolean, got {mask.dtype}"
    key_rand, _ = random.split(key)
    noise = random.uniform(key_rand, tuple(mask.shape), 0.0, 1.0, etl.float32)
    sorted_idx = etl.argsort(
        -(etl.cast(mask, etl.float32) + noise), axis=dim, stable=True
    )
    sliced_idx = sorted_idx[:count]
    # etl v1 scatter rejects scalar updates (rank-0 vs expected rank 1) —
    # use a rank-matching ones update instead (replacement semantics).
    return etl.scatter(
        enp.zeros(tuple(mask.shape), dtype=mask.dtype),
        sliced_idx,
        enp.ones(tuple(sliced_idx.shape), dtype=mask.dtype),
        axis=dim,
    )
