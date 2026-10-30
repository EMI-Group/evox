"""Sort helpers shared by the ES family, part of the 2.0 functional API.

Plain (non-`@evoxc.defn`) function: it may only be called inside an active
trace (a function passed to `evoxc.build`/`evoxc.evaluate`), since EvoX Compiler has no
eager mode.
"""

import evoxc

Tensor = evoxc.SymbolicTensor


def sort_by_key(keys: Tensor, population: Tensor) -> tuple[Tensor, Tensor]:
    """Sort `population` rows by ascending `keys`.

    Returns `(sorted_keys, sorted_population)`. `evoxc.gather` has numpy `take`
    semantics, so for a 1-D int64 index along axis 0 it equals `x[idx]`,
    exactly torch's `keys[order]` / `population[order]`.
    """
    order = evoxc.argsort(keys)  # int64 indices
    sorted_keys = evoxc.gather(keys, order, axis=0)
    sorted_population = evoxc.gather(population, order, axis=0)
    return sorted_keys, sorted_population
