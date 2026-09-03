import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

import evox_etl.algorithms.so.pso_variants.fs_pso as algo_mod

from unit_test.etl.algorithms.helpers import SphereConfig, run_generations


def test_fs_pso_smoke():
    """3 generations on 20-d Sphere: shapes stay (pop_size, dim), fitness finite."""
    cfg = algo_mod.FSPSO(
        pop_size=100,
        lb=np.full(20, -10.0, np.float32),
        ub=np.full(20, 10.0, np.float32),
    )
    state = run_generations(algo_mod, cfg, SphereConfig(20), n_gens=3, seed=0)

    assert state.pop.shape == (100, 20)
    assert state.fit.shape == (100,)
    assert np.isfinite(state.fit.numpy()).all()
    assert np.isfinite(state.global_best_fit.numpy())
