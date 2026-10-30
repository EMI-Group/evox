"""Smoke tests for the functional GuidedES port (no torch)."""

import numpy as np
from helpers import SphereConfig, run_generations

import evox.algorithms.so.es_variants.guided_es as guided_es


def test_guided_es_sphere():
    dim = 10
    cfg = guided_es.make_guided_es(
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
    cfg = guided_es.make_guided_es(
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


# --- guided subspace (regression: subspace_dims < dim used to raise) ---------
def test_guided_es_subspace_dims_smaller_than_dim():
    """With ``subspace_dims=3 < dim=10`` construction/init/step must run
    without a shape error (perturbations are sampled through the orthonormal
    basis of the stored gradient span) and the fitness must improve."""
    dim, subspace_dims = 10, 3
    cfg = guided_es.make_guided_es(
        pop_size=8,
        center_init=np.full((dim,), 5.0, np.float32),
        subspace_dims=subspace_dims,
        sigma=0.5,
        lr=1.0,
    )
    state1 = run_generations(guided_es, cfg, SphereConfig(dim), 1, seed=0)
    state4 = run_generations(guided_es, cfg, SphereConfig(dim), 4, seed=0)
    assert np.asarray(state4.grad_subspace.numpy()).shape == (subspace_dims, dim)
    assert np.asarray(state4.center.numpy()).shape == (dim,)
    assert np.asarray(state4.z.numpy()).shape == (cfg.pop_size, dim)
    assert np.all(np.isfinite(state4.center.numpy()))
    best1 = float(state1.best_fitness.numpy())
    best4 = float(state4.best_fitness.numpy())
    assert np.isfinite(best1) and np.isfinite(best4)
    assert best4 < best1
