"""Smoke test for the functional ETL PSO port.  ETL-only, no torch."""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "unit_test" / "etl" / "algorithms"))

import numpy as np

import evox_etl.algorithms.so.pso_variants.pso as pso_mod
from evox_etl.algorithms.so.pso_variants.pso import PSO
from helpers import SphereConfig, run_generations


def test_pso_smoke():
    """Run 3 generations of PSO on Sphere and check state shapes/finiteness."""
    cfg = PSO(
        pop_size=100,
        lb=np.full(20, -10.0, np.float32),
        ub=np.full(20, 10.0, np.float32),
    )
    state = run_generations(pso_mod, cfg, SphereConfig(20), n_gens=3, seed=0)

    assert state.pop.shape == (100, 20)
    assert state.velocity.shape == (100, 20)
    assert state.fit.shape == (100,)
    assert state.local_best_location.shape == (100, 20)
    assert state.local_best_fit.shape == (100,)
    assert state.global_best_location.shape == (20,)
    assert state.global_best_fit.shape == ()
    assert state.fit.dtype == np.dtype("float32")
    assert np.isfinite(state.fit.numpy()).all()
    assert np.isfinite(state.pop.numpy()).all()
    assert np.isfinite(float(state.global_best_fit.numpy()))
