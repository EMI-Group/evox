"""Smoke test for the functional ETL port of CMA-ES (no torch imports)."""

import numpy as np

import helpers
from helpers import SphereConfig, run_generations

import evox_etl.algorithms.so.es_variants.cma_es as cma_es


def test_cma_es_sphere():
    dim = 10
    cfg = cma_es.make_cma_es(mean_init=np.full((dim,), 5.0, np.float32), sigma=2.0)
    state = run_generations(cma_es, cfg, SphereConfig(dim), 3, seed=0)
    best = state.best_fitness.numpy()
    assert np.isfinite(best) and best < dim * 25
    assert np.asarray(state.mean.numpy()).shape == (1, dim)
    assert np.asarray(state.C.numpy()).shape == (dim, dim)
