"""Shared helpers for the pso_variants algorithm modules.

Plain-function ports of ``src/evox/algorithms/so/pso_variants/utils.py``
(read-only torch reference).  ETL has no eager mode, so these functions only
run inside an active trace.
"""

from typing import List, Tuple

import etl
import etl.numpy as enp

Tensor = etl.SymbolicTensor


def min_by(values: List[Tensor], keys: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """Find the value with the minimum key.

    Port of the torch evox ``min_by``: concatenate values and keys along axis
    0, take the argmin key index and gather the matching value/key back to
    their original shapes (etl has no ``index_select``/``squeeze``).
    """
    values = etl.concatenate(values, axis=0)
    keys = etl.concatenate(keys, axis=0)
    min_index = enp.reshape(etl.argmin(keys, axis=0), (1,))
    return (
        enp.reshape(etl.gather(values, min_index, axis=0), values.shape[1:]),
        enp.reshape(etl.gather(keys, min_index, axis=0), ()),
    )
