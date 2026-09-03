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

from evox_etl.algorithms._shim_selection_basic import (
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
    return winners_ts2, winners_full, winners_multi, pbest


@pytest.mark.parametrize("container", [list, tuple])
def test_shim_selection_basic(container):
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
    winners_ts2, winners_full, winners_multi, pbest = [o.numpy() for o in out]

    assert winners_ts2.shape == (N_ROUND,)
    assert np.issubdtype(winners_ts2.dtype, np.integer)
    assert np.all((winners_ts2 >= 0) & (winners_ts2 < POP))

    assert winners_full.shape == (N_ROUND,)
    assert np.all(winners_full == int(np.argmin(FITNESS)))

    assert winners_multi.shape == (N_ROUND,)
    assert np.all(winners_multi == 1)

    assert pbest.shape == (POP, DIM)
    src_idx = (pbest[:, 0] / 100.0).astype(np.int64)
    assert set(np.unique(src_idx)) <= {1, 2, 4}
    assert np.all(np.isin(PBF_FITNESS[src_idx], [0.0, 1.0, 2.0]))

    out2 = etl.run(exe, key, fitness, pbf_fitness, fitnesses, population)
    for a, b in zip(out, out2):
        assert np.array_equal(a.numpy(), b.numpy())
