"""Smoke tests for the functional xNES / SeparableNES ports (no torch).

Driven by ``run_generations`` from the shared helpers on the Sphere problem:
both algorithms must improve on a mean-5.0 / unit-covariance start within a
few generations.
"""

import math

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


# --- default pop_size (regression: it used to be unusable) -------------------
# ``make_xnes`` / ``make_separable_nes`` must derive ``pop_size`` (and the
# learning rates) from ``dim`` when it is not given: ``4 + floor(3 ln dim)``.


def test_xnes_default_pop_size_runs_and_improves():
    dim = 10
    cfg = nes.make_xnes(
        init_mean=np.full((dim,), 5.0, np.float32),
        init_covar=np.eye(dim, dtype=np.float32),
    )
    assert cfg.pop_size == 4 + math.floor(3 * math.log(dim)) == 10
    state1 = run_generations(nes, cfg, SphereConfig(dim=dim), n_gens=1, seed=42)
    state3 = run_generations(nes, cfg, SphereConfig(dim=dim), n_gens=3, seed=42)
    assert state3.mean.shape == (dim,) and state3.B.shape == (dim, dim)
    best1 = float(state1.best_fitness.numpy())
    best3 = float(state3.best_fitness.numpy())
    assert np.isfinite(best1) and np.isfinite(best3)
    assert best3 < best1  # fitness decreases from the mean-5.0 start (dim*25)


def test_separable_nes_default_pop_size_runs_and_improves():
    dim = 10
    cfg = nes.make_separable_nes(
        init_mean=np.full((dim,), 5.0, np.float32),
        init_std=np.full((dim,), 1.0, np.float32),
    )
    assert cfg.pop_size == 4 + math.floor(3 * math.log(dim)) == 10
    state1 = run_generations(nes, cfg, SphereConfig(dim=dim), n_gens=1, seed=42)
    state3 = run_generations(nes, cfg, SphereConfig(dim=dim), n_gens=3, seed=42)
    assert state3.mean.shape == (dim,) and state3.sigma.shape == (dim,)
    best1 = float(state1.best_fitness.numpy())
    best3 = float(state3.best_fitness.numpy())
    assert np.isfinite(best1) and np.isfinite(best3)
    assert best3 < best1
