"""Functional ETL ports of the torch evox basic selection operators.

Plain-function ports of ``src/evox/operators/selection/tournament_selection.py``
and ``src/evox/operators/selection/find_pbest.py`` (read-only torch reference).
ETL has no eager mode, so callers wrap these functions in a traced function via
``etl.build`` / ``etl.evaluate``.

Deviations from torch (mathematically equivalent):
- The RNG key is an explicit first argument (etl RNG is stateless); each
  ``torch.randint`` draw gets its own ``random.split`` subkey, in the same
  draw order with the same distribution.
- ``etl.gather`` has numpy ``take`` semantics, not torch ``gather``
  (``take_along_axis``); row-local selection uses ``_take_along_axis``.
- ``lexsort`` is inlined here as ``_lexsort`` (``evox_etl.algorithms._shim_utils``
  is a parallel task); exact port of the torch algorithm in
  ``src/evox/utils/jit_fix_operator.py`` lines 216-252. Note: like the torch
  original, that chain makes the LAST key the primary sort key (numpy lexsort
  convention); earlier keys only break ties among later keys (verified against
  torch).
"""

from typing import Sequence

import etl
import etl.numpy as enp
import etl.random as random
from etl.core import SymbolicTensor


def _take_along_axis(
    x: SymbolicTensor, indices: SymbolicTensor, axis: int
) -> SymbolicTensor:
    """Torch ``gather``/``take_along_axis`` semantics for 2-D ``x`` (etl lacks it).

    Selects ``out[i, j] = x[i, indices[i, j]]`` (axis=1) or
    ``out[i, j] = x[indices[i, j], j]`` (axis=0) by flattening the non-indexed
    axis into the offset, since ``etl.gather`` applies its index array to every
    row (numpy ``take`` semantics).
    """
    if axis == 1 and indices.shape[0] != x.shape[0]:
        raise ValueError(
            f"take_along_axis: indices leading dim {indices.shape[0]} must match "
            f"x axis 0 size {x.shape[0]} for row-aligned selection along axis 1"
        )
    r0, r1 = x.shape
    flat = enp.reshape(x, (r0 * r1,))
    if axis == 0:
        off = enp.reshape(
            etl.arange(indices.shape[1], dtype=etl.int64) * r0, (1, indices.shape[1])
        )
    else:
        off = enp.reshape(
            etl.arange(indices.shape[0], dtype=etl.int64) * r1, (indices.shape[0], 1)
        )
    flat_idx = etl.cast(indices, etl.int64) + off
    return enp.reshape(etl.gather(flat, flat_idx, axis=0), indices.shape)


def _lexsort(keys: Sequence[SymbolicTensor], dim: int = -1) -> SymbolicTensor:
    """Lexicographic sort indices, exact port of the torch evox ``lexsort``.

    Stable-argsort chain from ``src/evox/utils/jit_fix_operator.py`` lines
    216-252: sort by ``keys[0]``, then refine with each later key. Like the
    torch original, the last key ends up primary (numpy lexsort convention),
    with earlier keys as tiebreakers.
    """
    axis = dim % len(keys[0].shape)
    sorted_indices = etl.argsort(keys[0], axis=axis, stable=True)
    for key in keys[1:]:
        sorted_key = _take_along_axis(key, sorted_indices, axis)
        final_sorted_indices = etl.argsort(sorted_key, axis=axis, stable=True)
        sorted_indices = _take_along_axis(sorted_indices, final_sorted_indices, axis)
    return sorted_indices


def tournament_selection_multifit(
    key: SymbolicTensor,
    n_round: int,
    fitnesses: Sequence[SymbolicTensor],
    tournament_size: int = 2,
) -> SymbolicTensor:
    """Tournament selection over multiple fitness values (lexsort winner).

    Port of ``evox.operators.selection.tournament_selection_multifit``: draws
    ``n_round`` tournaments and returns the first lexsort element of each
    (last objective primary, earlier objectives as tiebreakers — torch 1:1).
    """
    fitness_tensor = etl.stack(fitnesses, axis=1)
    num_candidates = fitness_tensor.shape[0]

    key_rand, _ = random.split(key)
    parents = random.randint(
        key_rand, (n_round, tournament_size), 0, num_candidates, dtype=etl.int32
    )
    candidates_fitness = etl.gather(fitness_tensor, parents, axis=0)
    candidates_fitness = _lexsort(
        [candidates_fitness[:, :, i] for i in range(fitness_tensor.shape[1])]
    )

    selected_parents = _take_along_axis(
        parents, enp.reshape(candidates_fitness[:, 0], (n_round, 1)), axis=1
    )
    return enp.reshape(selected_parents, (n_round,))


def tournament_selection(
    key: SymbolicTensor,
    n_round: int,
    fitness: SymbolicTensor,
    tournament_size: int = 2,
) -> SymbolicTensor:
    """Tournament selection on a single fitness value (winner = min fitness).

    Port of ``evox.operators.selection.tournament_selection``.
    """
    num_candidates = fitness.shape[0]

    key_rand, _ = random.split(key)
    parents = random.randint(
        key_rand, (n_round, tournament_size), 0, num_candidates, dtype=etl.int32
    )
    candidates_fitness = etl.gather(fitness, parents, axis=0)

    winner_indices = etl.argmin(candidates_fitness, axis=1)

    selected_parents = _take_along_axis(
        parents, enp.reshape(winner_indices, (n_round, 1)), axis=1
    )
    return enp.reshape(selected_parents, (n_round,))


def select_rand_pbest(
    key: SymbolicTensor,
    percent: float,
    population: SymbolicTensor,
    fitness: SymbolicTensor,
) -> SymbolicTensor:
    """Random personal-best selection from the top ``percent`` of the population.

    Port of ``evox.operators.selection.find_pbest.select_rand_pbest``.
    """
    pop_size = population.shape[0]
    top_p_num = max(int(pop_size * percent), 1)
    pbest_indices_pool = etl.argsort(fitness, axis=0)[:top_p_num]

    key_rand, _ = random.split(key)
    random_indices = random.randint(
        key_rand, (pop_size,), 0, top_p_num, dtype=etl.int32
    )
    pool_picks = etl.gather(pbest_indices_pool, random_indices, axis=0)
    return etl.gather(population, pool_picks, axis=0)
