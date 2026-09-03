"""Utility helpers shared by the PSO variants (functional ETL ports).

NOTE: this file is also written by a parallel agent (full version). This
minimal copy provides ``min_by`` so the fs_pso port is self-contained; if a
merge conflict arises, the manager resolves it.
"""

from typing import Any, List, Tuple

import etl
import etl.numpy as enp


def min_by(values: List[Any], keys: List[Any]) -> Tuple[Any, Any]:
    """Find the value with the minimum key (1:1 port of the torch evox ``min_by``).

    Concatenates along axis 0, argmins the keys and gathers the winning row.
    """
    values = etl.concatenate(values, axis=0)
    keys = etl.concatenate(keys, axis=0)
    min_index = etl.argmin(keys, axis=0)
    min_index = enp.reshape(min_index, (1,))
    return (
        enp.reshape(etl.gather(values, min_index, axis=0), values.shape[1:]),
        enp.reshape(etl.gather(keys, min_index, axis=0), keys.shape[1:]),
    )
