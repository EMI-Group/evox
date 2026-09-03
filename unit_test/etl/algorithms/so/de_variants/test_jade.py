"""Smoke test for the functional JaDE port (no torch)."""
import sys
from pathlib import Path

sys.path[0:0] = [str(Path(__file__).resolve().parents[5]), str(Path(__file__).resolve().parents[5] / "src")]
import numpy as np

import evox_etl.algorithms.so.de_variants.jade as jade_mod
from evox_etl.algorithms.so.de_variants.jade import JaDE
from unit_test.etl.algorithms.helpers import SphereConfig, run_generations

POP, DIM, GENS = 10, 20, 3


def test_jade_smoke():
    cfg = JaDE(pop_size=POP, lb=np.full(DIM, -10.0), ub=np.full(DIM, 10.0))
    state = run_generations(jade_mod, cfg, SphereConfig(DIM), GENS, seed=0)
    fit = state.fit.numpy()
    assert fit.shape == (POP,) and np.all(np.isfinite(fit))
    pop = state.pop.numpy()
    assert pop.shape == (POP, DIM)
    assert np.all(pop >= -10.0) and np.all(pop <= 10.0)
    fu, cru = state.F_u.numpy(), state.CR_u.numpy()
    assert np.all(np.isfinite(fu)) and np.all((fu >= 0.0) & (fu <= 1.0))
    assert np.all(np.isfinite(cru)) and np.all((cru >= 0.0) & (cru <= 1.0))
