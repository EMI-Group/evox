"""Smoke tests for the functional ETL port of ESMC (no torch)."""

import numpy as np

import helpers
from helpers import SphereConfig, run_generations

import evox_etl.algorithms.so.es_variants.esmc as esmc


def _cfg(optimizer):
    dim = 10
    return esmc.ESMCConfig(
        pop_size=9,
        center_init=np.full((dim,), 5.0, np.float32),
        optimizer=optimizer,
        sigma=0.5,
        lr=0.1,
    )


def test_esmc_sphere():
    dim = 10
    cfg = _cfg(None)
    state = run_generations(esmc, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)


def test_esmc_sphere_adam():
    dim = 10
    cfg = _cfg("adam")
    state = run_generations(esmc, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)
