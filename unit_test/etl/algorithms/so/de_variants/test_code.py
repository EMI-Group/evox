"""Smoke test for the functional CoDE port (no torch)."""
import sys
from pathlib import Path
sys.path[0:0] = [str(Path(__file__).resolve().parents[5]), str(Path(__file__).resolve().parents[5] / "src")]
import numpy as np
import evox_etl.algorithms.so.de_variants.code as code_mod
from evox_etl.algorithms.so.de_variants.code import CoDE
from unit_test.etl.algorithms.helpers import SphereConfig, run_generations
POP, DIM, GENS = 10, 20, 3
def test_code_smoke():
    cfg = CoDE(pop_size=POP, lb=np.full(DIM, -10.0), ub=np.full(DIM, 10.0))
    state = run_generations(code_mod, cfg, SphereConfig(DIM), GENS, seed=0)
    fit = state.fit.numpy()
    assert fit.shape == (POP,) and np.all(np.isfinite(fit))
    assert state.pop.numpy().shape == (POP, DIM)
