"""Smoke tests for the functional OpenES port (ETL only — no torch)."""

import numpy as np

import helpers
from helpers import SphereConfig, run_generations
import evox_etl.algorithms.so.es_variants.open_es as open_es


def _cfg(optimizer=None):
    dim = 10
    return dim, open_es.OpenESConfig(
        pop_size=8,
        center_init=np.full((dim,), 5.0, np.float32),
        learning_rate=0.01,
        noise_stdev=1.0,
        optimizer=optimizer,
    )


def test_open_es_sphere():
    dim, cfg = _cfg()
    state = run_generations(open_es, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    # Sphere at center=5 has fitness dim*25; 3 generations must improve.
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)


def test_open_es_adam_sphere():
    dim, cfg = _cfg(optimizer="adam")
    state = run_generations(open_es, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)
