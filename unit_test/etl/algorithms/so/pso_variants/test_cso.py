"""Smoke tests for the functional CSO port (ETL-only, no torch)."""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "unit_test" / "etl" / "algorithms"))

import numpy as np

from helpers import SphereConfig, run_generations

import evox_etl.algorithms.so.pso_variants.cso as cso
from evox_etl.algorithms.so.pso_variants.cso import CSO


def _base_cfg() -> CSO:
    return CSO(
        pop_size=100,
        lb=np.full(20, -10.0, np.float32),
        ub=np.full(20, 10.0, np.float32),
    )


def test_cso_smoke():
    state = run_generations(cso, _base_cfg(), SphereConfig(20), n_gens=3, seed=0)
    assert state.pop.shape == (100, 20)
    assert state.fit.shape == (100,)
    assert np.isfinite(state.fit.numpy()).all()
    assert state.students.shape == (50,)


def test_cso_mean_stdev_init():
    cfg = CSO(
        pop_size=100,
        lb=np.full(20, -10.0, np.float32),
        ub=np.full(20, 10.0, np.float32),
        mean=np.zeros(20, np.float32),
        stdev=np.full(20, 0.5, np.float32),
    )
    state = run_generations(cso, cfg, SphereConfig(20), n_gens=2, seed=1)
    assert state.pop.shape == (100, 20)
    assert np.isfinite(state.pop.numpy()).all()
    assert np.isfinite(state.fit.numpy()).all()
