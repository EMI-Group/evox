"""Smoke tests for the functional ETL port of DMS-PSO-EL (no torch).

Driven through ``run_generations`` from the sibling helpers module: the gen-0
loop iteration uses ``init_ask``/``init_tell`` (the init generation — torch
``init_step`` increments the counter there), later iterations use ``ask``/
``tell`` (each ask-increment counts one generation).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "unit_test" / "etl" / "algorithms"))

import numpy as np

import evox_etl.algorithms.so.pso_variants.dms_pso_el as dms_pso_el
from helpers import SphereConfig, run_generations

DIM = 20
LB = np.full(DIM, -10.0, np.float32)
UB = np.full(DIM, 10.0, np.float32)
DSS, DSSN, FSS = 10, 5, 10
POP_SIZE = DSS * DSSN + FSS


def _make_config(**kwargs):
    kwargs.setdefault("dynamic_sub_swarm_size", DSS)
    kwargs.setdefault("dynamic_sub_swarms_num", DSSN)
    kwargs.setdefault("following_sub_swarm_size", FSS)
    return dms_pso_el.make_dms_pso_el(lb=LB, ub=UB, **kwargs)


def _assert_common(state):
    assert state.pop.shape == (POP_SIZE, DIM)
    assert state.velocity.shape == (POP_SIZE, DIM)
    assert state.fit.shape == (POP_SIZE,)
    assert np.isfinite(state.fit.numpy()).all()
    assert state.personal_best_fit.shape == (POP_SIZE,)
    assert np.isfinite(state.personal_best_fit.numpy()).all()
    assert state.local_best_location.shape == (DSSN, DIM)
    assert state.regional_best_index.shape == (FSS,)
    assert state.global_best_location.shape == (DIM,)


def test_default_regroup_identity_path():
    """Default config: 3 generations on Sphere, strategy 1 without regrouping."""
    state = run_generations(
        dms_pso_el,
        _make_config(),
        SphereConfig(DIM),
        n_gens=3,
        seed=0,
    )
    _assert_common(state)
    # gen0 = init generation (init_tell: 0 -> 1); gens 1-2 ask-increment -> 3.
    assert int(np.asarray(state.iteration.numpy())) == 3


def test_regroup_every_step():
    """regrouped_iteration_num=1 forces _regroup inside every strategy-1 ask."""
    state = run_generations(
        dms_pso_el,
        _make_config(regrouped_iteration_num=1),
        SphereConfig(DIM),
        n_gens=2,
        seed=0,
    )
    _assert_common(state)
    # gen0 = init generation (0 -> 1); gen1 regroup-ask (1 -> 2).
    assert int(np.asarray(state.iteration.numpy())) == 2


def test_strategy2_path():
    """max_iteration=3: the ask with iteration 3 >= 0.9*3 runs strategy 2,
    which is the only path that updates global_best_fit (init is +inf)."""
    state = run_generations(
        dms_pso_el,
        _make_config(max_iteration=3),
        SphereConfig(DIM),
        n_gens=4,
        seed=0,
    )
    _assert_common(state)
    assert np.isfinite(state.global_best_fit.numpy())
