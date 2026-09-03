"""Smoke test for the functional ETL port of SLPSOGS (no torch)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "unit_test" / "etl" / "algorithms"))

import numpy as np

from helpers import SphereConfig, run_generations

from evox_etl.algorithms.so.pso_variants import sl_pso_gs


def test_sl_pso_gs_smoke():
    """3 generations on 20-D Sphere: correct shapes and finite fitnesses."""
    cfg = sl_pso_gs.SLPSOGS(
        pop_size=100,
        lb=np.full(20, -10.0, np.float32),
        ub=np.full(20, 10.0, np.float32),
    )
    state = run_generations(sl_pso_gs, cfg, SphereConfig(20), n_gens=3, seed=0)
    assert state.pop.shape == (100, 20)
    assert state.fit.shape == (100,)
    assert np.isfinite(np.asarray(state.fit.numpy())).all()
    assert np.isfinite(float(state.global_best_fit.numpy()))
