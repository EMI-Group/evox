"""Smoke test for the functional ODE port (no torch)."""
import sys
from pathlib import Path

sys.path[0:0] = [
    str(Path(__file__).resolve().parents[5]),
    str(Path(__file__).resolve().parents[5] / "src"),
]
import numpy as np

import evox_etl.algorithms.so.de_variants.ode as ode_mod
from evox_etl.algorithms.so.de_variants.ode import ODE
from unit_test.etl.algorithms.helpers import SphereConfig, run_generations

POP, DIM, GENS = 10, 20, 4


def _cfg(**kw):
    return ODE(pop_size=POP, lb=np.full(DIM, -10.0), ub=np.full(DIM, 10.0), **kw)


def test_ode_rand_smoke():
    state = run_generations(ode_mod, _cfg(base_vector="rand"), SphereConfig(DIM), GENS, seed=0)
    fit = state.fit.numpy()
    assert fit.shape == (POP,) and np.all(np.isfinite(fit))
    pop = state.pop.numpy()
    assert pop.shape == (POP, DIM)
    assert np.all(pop >= -10.0) and np.all(pop <= 10.0)


def test_ode_best_smoke():
    state = run_generations(ode_mod, _cfg(base_vector="best"), SphereConfig(DIM), GENS, seed=0)
    assert np.all(np.isfinite(state.fit.numpy()))
