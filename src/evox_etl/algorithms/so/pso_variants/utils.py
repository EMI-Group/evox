"""Shared helpers for the PSO variant algorithms.

Temporary minimal module providing `min_by` only — ported 1:1 from the torch
``src/evox/algorithms/so/pso_variants/utils.py``. The full
``evox_etl.utils`` package is being built in parallel; when it lands this
file may be merged into it (the `min_by` implementation below is
authoritative).
"""

from typing import Any, List, Tuple

import etl
import etl.numpy as enp

__all__ = ["min_by"]


def min_by(values: List[Any], keys: List[Any]) -> Tuple[Any, Any]:
    """Return the value with the minimum key.

    Concatenates ``values`` and ``keys`` along axis 0, then selects the
    value/key at the argmin position (torch `min_by` semantics).
    """
    values = etl.concatenate(values, axis=0)
    keys = etl.concatenate(keys, axis=0)
    min_index = enp.reshape(etl.argmin(keys, axis=0), (1,))
    value = enp.reshape(etl.gather(values, min_index, axis=0), values.shape[1:])
    key = enp.reshape(etl.gather(keys, min_index, axis=0), keys.shape[1:])
    return value, key
