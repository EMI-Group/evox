"""Tests for the canonical basic selection operators in ``evox_etl.operators.selection``.

Converted from the deprecated ``evox_etl.algorithms._shim_selection_basic``
compat stub (which split the key once before delegating to the canonical
operators). The canonical operators draw directly from the key they are given
(handling their own internal splits), so the old shim draw-stream convention is
gone: the shim's fixed key-split produced a lucky stream where full-tournament
rounds always drew the global best; the canonical stream does not, so
draw-stream-dependent winner assertions were replaced by exact-value checks
that replicate the internal draws (keys are pure: same key + same parameters
-> identical draws, so the replication is exact) plus deterministic replay
with the same inputs.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pytest

import etl
import etl.random as random
from etl.core import TensorSpec, from_numpy

from evox_etl.operators.selection import (
    select_rand_pbest,
    tournament_selection,
    tournament_selection_multifit,
)

POP = 6
N_ROUND = 7
DIM = 3

FITNESS = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 0.0], dtype=np.float32)
FITNESS_0 = np.array([3.0, 1.0, 2.0, 4.0, 1.0, 6.0], dtype=np.float32)
FITNESS_1 = np.array([9.0, 0.0, 8.0, 0.0, 7.0, 6.0], dtype=np.float32)
PBF_FITNESS = np.array([4.0, 2.0, 0.0, 3.0, 1.0, 5.0], dtype=np.float32)
POPULATION = (
    100.0 * np.arange(POP, dtype=np.float32)[:, None]
    + np.arange(DIM, dtype=np.float32)[None, :]
)

KEY_SPEC = TensorSpec(shape=(), dtype=np.dtype("int64"))
FIT_SPEC = TensorSpec(shape=(POP,), dtype=np.dtype("float32"))
POP_SPEC = TensorSpec(shape=(POP, DIM), dtype=np.dtype("float32"))

SEED = 1026


def _exercise_all(k, fitness, pbf_fitness, fitnesses, population):
    k1, k2 = random.split(k)
    k3, k4 = random.split(k2)
    winners_ts2 = tournament_selection(k1, N_ROUND, fitness)
    winners_full = tournament_selection(k2, N_ROUND, fitness, tournament_size=POP)
    winners_multi = tournament_selection_multifit(k3, N_ROUND, fitnesses, tournament_size=POP)
    pbest = select_rand_pbest(k4, 0.5, population, pbf_fitness)
    # Replicate the internal draws with the same keys (keys are pure, so these
    # are bit-identical to the draws inside the operators) to reconstruct the
    # expected winners in numpy.
    parents_ts2 = random.randint(k1, (N_ROUND, 2), 0, POP, 'int32')
    parents_full = random.randint(k2, (N_ROUND, POP), 0, POP, 'int32')
    parents_multi = random.randint(k3, (N_ROUND, POP), 0, POP, 'int32')
    pbest_pool = etl.argsort(pbf_fitness)[:3]
    pbest_idx = random.randint(k4, (POP,), 0, 3, 'int32')
    return (
        winners_ts2, winners_full, winners_multi, pbest,
        parents_ts2, parents_full, parents_multi, pbest_pool, pbest_idx,
    )


@pytest.mark.parametrize("container", [list, tuple])
def test_selection_basic(container):
    exe = etl.build(
        _exercise_all,
        KEY_SPEC, FIT_SPEC, FIT_SPEC,
        container([FIT_SPEC, FIT_SPEC]), POP_SPEC,
        backend="numpy",
    )

    key = from_numpy(np.array(SEED, dtype=np.int64))
    fitness = from_numpy(FITNESS)
    pbf_fitness = from_numpy(PBF_FITNESS)
    fitnesses = container([from_numpy(FITNESS_0), from_numpy(FITNESS_1)])
    population = from_numpy(POPULATION)

    out = etl.run(exe, key, fitness, pbf_fitness, fitnesses, population)
    (winners_ts2, winners_full, winners_multi, pbest,
     parents_ts2, parents_full, parents_multi, pbest_pool, pbest_idx) = [
        o.numpy() for o in out
    ]

    assert winners_ts2.shape == (N_ROUND,)
    assert np.issubdtype(winners_ts2.dtype, np.integer)
    assert np.all((winners_ts2 >= 0) & (winners_ts2 < POP))

    # exact winners: per round, the drawn candidate with the minimal fitness
    exp_ts2 = np.array(
        [parents_ts2[r, np.argmin(FITNESS[parents_ts2[r]])] for r in range(N_ROUND)]
    )
    np.testing.assert_array_equal(winners_ts2, exp_ts2)

    assert winners_full.shape == (N_ROUND,)
    exp_full = np.array(
        [parents_full[r, np.argmin(FITNESS[parents_full[r]])] for r in range(N_ROUND)]
    )
    np.testing.assert_array_equal(winners_full, exp_full)

    # multifit: numpy lexsort (last key primary, matches the etl port)
    assert winners_multi.shape == (N_ROUND,)
    exp_multi = np.array(
        [
            parents_multi[r, np.lexsort((FITNESS_0[parents_multi[r]],
                                         FITNESS_1[parents_multi[r]]))[0]]
            for r in range(N_ROUND)
        ]
    )
    np.testing.assert_array_equal(winners_multi, exp_multi)

    # pbest: pool is the 3 best by fitness; drawn index picks the source row
    assert pbest.shape == (POP, DIM)
    src_idx = (pbest[:, 0] / 100.0).astype(np.int64)
    assert set(np.unique(src_idx)) <= {1, 2, 4}
    assert np.all(np.isin(PBF_FITNESS[src_idx], [0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(pbest_pool, np.array([2, 4, 1], dtype=np.int32))
    exp_pbest = POPULATION[pbest_pool[pbest_idx]]
    np.testing.assert_array_equal(pbest, exp_pbest)

    out2 = etl.run(exe, key, fitness, pbf_fitness, fitnesses, population)
    for a, b in zip(out, out2):
        assert np.array_equal(a.numpy(), b.numpy())
