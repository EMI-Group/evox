import sys
from pathlib import Path
sys.path[0:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[3] / "src")]

import itertools
from math import comb

import numpy as np
import etl
from etl import core

from evox_etl.algorithms._shim_mutation_sampling import polynomial_mutation, uniform_sampling


def _exercise(key, x, boundary):
    pm = polynomial_mutation(key, x, boundary)
    w, n_samples = uniform_sampling(20, 3)
    return pm, w, n_samples


def das_dennis_reference(n, m):
    h = 1
    while comb(h + m, m - 1) <= n:
        h += 1
    w = (np.array(list(itertools.combinations(range(1, h + m), m - 1)), dtype=np.float32)
         - np.tile(np.arange(m - 1, dtype=np.float32), (comb(h + m - 1, m - 1), 1)) - 1)
    w = (np.concatenate([w, np.zeros((w.shape[0], 1), dtype=np.float32) + h], axis=1)
         - np.concatenate([np.zeros((w.shape[0], 1), dtype=np.float32), w], axis=1)) / h
    if h < m:
        h2 = 0
        while comb(h + m - 1, m - 1) + comb(h2 + m, m - 1) <= n:
            h2 += 1
        if h2 > 0:
            w2 = (np.array(list(itertools.combinations(range(1, h2 + m), m - 1)), dtype=np.float32)
                  - np.tile(np.arange(m - 1, dtype=np.float32), (comb(h2 + m - 1, m - 1), 1)) - 1)
            w2 = (np.concatenate([w2, np.zeros((w2.shape[0], 1), dtype=np.float32) + h2], axis=1)
                  - np.concatenate([np.zeros((w2.shape[0], 1), dtype=np.float32), w2], axis=1)) / h2
            w = np.concatenate([w, w2 / 2.0 + 1.0 / (2.0 * m)], axis=0)
    return np.maximum(w, 1e-6)


class TestShimMutationSampling:
    def run(self):
        key = np.array(7, dtype=np.int64)
        x = np.array([[0.5, -1.5, 3.0, 0.2],
                      [2.5, 0.0, -2.0, 1.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [1.0, -0.5, 0.5, -1.0],
                      [-2.0, 2.0, 1.5, 0.75],
                      [0.25, 0.5, -0.25, -0.75]], dtype=np.float32)
        boundary = np.array([-1.0, 1.0], dtype=np.float32)
        exe = etl.build(
            _exercise,
            core.TensorSpec(shape=(), dtype=np.dtype("int64")),
            core.TensorSpec(shape=(6, 4), dtype=np.dtype("float32")),
            core.TensorSpec(shape=(2,), dtype=np.dtype("float32")),
            backend="numpy",
        )
        outs = etl.run(exe, key, x, boundary)
        return x, boundary, [o.numpy() for o in outs[:2]], outs[2]

    def test_polynomial_mutation(self):
        x, boundary, (pm, _w), _ = self.run()
        assert pm.shape == x.shape and pm.dtype == np.float32
        assert np.isfinite(pm).all()
        assert np.all(pm >= boundary[0] - 1e-6) and np.all(pm <= boundary[1] + 1e-6)

    def test_uniform_sampling(self):
        _x, _b, (_pm, w), n_samples = self.run()
        ref = das_dennis_reference(20, 3)
        assert isinstance(n_samples, int)
        assert n_samples == ref.shape[0]
        assert w.shape == (n_samples, 3) and w.dtype == np.float32
        np.testing.assert_allclose(w, ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(w.sum(axis=1), np.ones(n_samples), rtol=1e-5, atol=1e-5)

    def test_determinism(self):
        x, boundary, (pm1, w1), _ = self.run()
        pm2, w2 = self.run()[2]
        assert np.array_equal(pm1, pm2) and np.array_equal(w1, w2)
