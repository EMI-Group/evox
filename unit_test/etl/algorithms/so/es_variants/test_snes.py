"""Smoke tests for the functional ETL port of SNES (no torch imports)."""
import numpy as np
import helpers
from helpers import SphereConfig, run_generations
import evox_etl.algorithms.so.es_variants.snes as snes


def test_snes_temp_sphere():
    dim = 10
    cfg = snes.SNESConfig(pop_size=8, center_init=np.full((dim,), 5.0, np.float32), sigma=1.0, weight_type="temp")
    state = run_generations(snes, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)
    assert np.asarray(state.sigma.numpy()).shape == (dim,)


def test_snes_recomb_sphere():
    dim = 10
    cfg = snes.SNESConfig(pop_size=8, center_init=np.full((dim,), 5.0, np.float32), sigma=1.0, weight_type="recomb")
    state = run_generations(snes, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)
    assert np.asarray(state.sigma.numpy()).shape == (dim,)
