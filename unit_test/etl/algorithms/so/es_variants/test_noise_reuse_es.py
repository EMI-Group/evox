"""Smoke tests for the functional NoiseReuseES port (no torch)."""

import numpy as np

import helpers
from helpers import SphereConfig, run_generations

import evox_etl.algorithms.so.es_variants.noise_reuse_es as noise_reuse_es


def test_noise_reuse_es_sphere():
    dim = 10
    cfg = noise_reuse_es.make_noise_reuse_es(
        pop_size=8,
        center_init=np.full((dim,), 5.0, np.float32),
        sigma=0.5,
        lr=0.1,
        T=2,
        K=1,
    )
    state = run_generations(noise_reuse_es, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)


def test_noise_reuse_es_sphere_adam():
    dim = 10
    cfg = noise_reuse_es.make_noise_reuse_es(
        pop_size=8,
        center_init=np.full((dim,), 5.0, np.float32),
        optimizer="adam",
        sigma=0.5,
        lr=0.1,
        T=2,
        K=1,
    )
    state = run_generations(noise_reuse_es, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.center.numpy()).shape == (dim,)
