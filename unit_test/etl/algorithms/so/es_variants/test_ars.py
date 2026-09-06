"""Smoke tests for the functional ARS port (ETL only — no torch)."""

import numpy as np

import helpers
from helpers import SphereConfig, run_generations
import evox_etl.algorithms.so.es_variants.ars as ars


def _cfg(optimizer=None):
    dim = 10
    return dim, ars.make_ars(
        pop_size=8,
        center_init=np.full((dim,), 5.0, np.float32),
        elite_ratio=0.5,
        lr=0.05,
        sigma=0.5,
        optimizer=optimizer,
    )


def test_ars_sphere():
    dim, cfg = _cfg()
    state = run_generations(ars, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    # Sphere at center=5 has fitness dim*25; 3 generations must improve.
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)


def test_ars_adam_sphere():
    dim, cfg = _cfg(optimizer="adam")
    state = run_generations(ars, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)
