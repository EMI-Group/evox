"""Smoke tests for the functional ETL port of DES (no torch imports)."""
import numpy as np
import helpers
from helpers import SphereConfig, run_generations
import evox_etl.algorithms.so.es_variants.des as des


def test_des_sphere():
    dim = 10
    cfg = des.DESConfig(pop_size=8, center_init=np.full((dim,), 5.0, np.float32), sigma_init=0.5)
    state = run_generations(des, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)
    assert np.asarray(state.sigma.numpy()).shape == (dim,)
