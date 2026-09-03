"""Smoke tests for the functional GuidedES port (no torch)."""

import numpy as np

import helpers
from helpers import SphereConfig, run_generations

import evox_etl.algorithms.so.es_variants.guided_es as guided_es


def test_guided_es_sphere():
    dim = 10
    cfg = guided_es.GuidedESConfig(
        pop_size=8,
        center_init=np.full((dim,), 5.0, np.float32),
        sigma=0.5,
        lr=1.0,
    )
    state = run_generations(guided_es, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)


def test_guided_es_sphere_adam():
    dim = 10
    cfg = guided_es.GuidedESConfig(
        pop_size=8,
        center_init=np.full((dim,), 5.0, np.float32),
        optimizer="adam",
        sigma=0.5,
        lr=1.0,
    )
    state = run_generations(guided_es, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)
