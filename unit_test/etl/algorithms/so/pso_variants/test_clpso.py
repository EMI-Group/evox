"""Smoke tests for the functional CLPSO port (ETL-only, no torch)."""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "unit_test" / "etl" / "algorithms"))

import numpy as np

from helpers import SphereConfig, run_generations

import evox_etl.algorithms.so.pso_variants.clpso as clpso
from evox_etl.algorithms.so.pso_variants.clpso import CLPSO


def _base_cfg() -> CLPSO:
    return CLPSO(
        pop_size=100,
        lb=np.full(20, -10.0, np.float32),
        ub=np.full(20, 10.0, np.float32),
    )


def test_clpso_smoke():
    state = run_generations(clpso, _base_cfg(), SphereConfig(20), n_gens=3, seed=0)
    assert state.pop.shape == (100, 20)
    assert state.fit.shape == (100,)
    assert np.isfinite(state.fit.numpy()).all()
    assert state.personal_best_location.shape == (100, 20)
    assert state.personal_best_fit.shape == (100,)
    assert state.global_best_location.shape == (20,)
    assert state.global_best_fit.shape == ()
    assert np.isfinite(state.global_best_fit.numpy()).all()
