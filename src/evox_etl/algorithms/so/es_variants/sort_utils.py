"""Functional ETL port of the torch evox `sort_utils.py` helper.

Plain (non-`@etl.defn`) function: it may only be called inside an active
trace (a function passed to `etl.build`/`etl.evaluate`), since ETL has no
eager mode. Semantics mirror the torch original in
`src/evox/algorithms/so/es_variants/sort_utils.py` exactly.
"""

import etl

Tensor = etl.SymbolicTensor


def sort_by_key(keys: Tensor, population: Tensor) -> tuple[Tensor, Tensor]:
    """Sort `population` rows by ascending `keys`.

    Returns `(sorted_keys, sorted_population)`. `etl.gather` has numpy `take`
    semantics, so for a 1-D int64 index along axis 0 it equals `x[idx]`,
    exactly torch's `keys[order]` / `population[order]`.
    """
    order = etl.argsort(keys)  # int64 indices
    sorted_keys = etl.gather(keys, order, axis=0)
    sorted_population = etl.gather(population, order, axis=0)
    return sorted_keys, sorted_population
