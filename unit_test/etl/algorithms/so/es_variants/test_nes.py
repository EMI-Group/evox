"""Smoke tests for the functional xNES / SeparableNES ports (no torch).

Driven by ``run_generations`` from the shared helpers on the Sphere problem:
both algorithms must improve on a mean-5.0 / unit-covariance start within a
few generations.
"""

import numpy as np

import helpers
from helpers import SphereConfig, run_generations
import evox_etl.algorithms.so.es_variants.nes as nes


def test_xnes_sphere():
    dim = 10
    cfg = nes.make_xnes(
        pop_size=8,
        init_mean=np.full((dim,), 5.0, np.float32),
        init_covar=np.eye(dim, dtype=np.float32),
    )
    state = run_generations(nes, cfg, SphereConfig(dim=dim), n_gens=3, seed=42)
    assert state.mean.shape == (dim,)
    assert state.sigma.shape == ()  # scalar step size for xNES
    assert float(state.best_fitness.numpy()) < dim * 25


def test_separable_nes_sphere():
    dim = 10
    cfg = nes.make_separable_nes(
        pop_size=8,
        init_mean=np.full((dim,), 5.0, np.float32),
        init_std=np.full((dim,), 1.0, np.float32),
    )
    state = run_generations(nes, cfg, SphereConfig(dim=dim), n_gens=3, seed=42)
    assert state.mean.shape == (dim,)
    assert state.sigma.shape == (dim,)  # per-dimension step sizes
    assert float(state.best_fitness.numpy()) < dim * 25
