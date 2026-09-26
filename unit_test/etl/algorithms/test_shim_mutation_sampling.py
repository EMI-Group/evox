"""Tests for the canonical mutation/sampling operators.

Converted from the deprecated ``evox_etl.algorithms._shim_mutation_sampling``
compat stub: ``polynomial_mutation`` now uses the canonical
``polynomial_mutation(key, x, lb, ub, pro_m=1.0, dis_m=20.0)`` signature with
separate lower/upper bound tensors instead of the old stacked ``boundary``.
"""
import sys
from pathlib import Path
sys.path[0:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[3] / "src")]

import itertools
from math import comb

import numpy as np
import etl
from etl import core

from evox_etl.operators.mutation import polynomial_mutation
from evox_etl.operators.sampling import uniform_sampling


def _exercise(key, x, lb, ub):
    # pro_m/dis_m stay at the python defaults baked at trace time (per etl
    # conventions they are NOT passed to etl.build/etl.run).
    pm = polynomial_mutation(key, x, lb, ub)
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


class TestMutationSamplingOperators:
    def run(self):
        key = np.array(7, dtype=np.int64)
        x = np.array([[0.5, -1.5, 3.0, 0.2],
                      [2.5, 0.0, -2.0, 1.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [1.0, -0.5, 0.5, -1.0],
                      [-2.0, 2.0, 1.5, 0.75],
                      [0.25, 0.5, -0.25, -0.75]], dtype=np.float32)
        # canonical signature: separate lb/ub tensors (0-d scalars broadcast
        # against x like the old stacked boundary [-1, 1]).
        lb = np.array(-1.0, dtype=np.float32)
        ub = np.array(1.0, dtype=np.float32)
        exe = etl.build(
            _exercise,
            core.TensorSpec(shape=(), dtype=np.dtype("int64")),
            core.TensorSpec(shape=(6, 4), dtype=np.dtype("float32")),
            core.TensorSpec(shape=(), dtype=np.dtype("float32")),
            core.TensorSpec(shape=(), dtype=np.dtype("float32")),
            backend="numpy",
        )
        outs = etl.run(exe, key, x, lb, ub)
        return x, lb, ub, [o.numpy() for o in outs[:2]], outs[2]

    def test_polynomial_mutation(self):
        x, lb, ub, (pm, _w), _ = self.run()
        assert pm.shape == x.shape and pm.dtype == np.float32
        assert np.isfinite(pm).all()
        assert np.all(pm >= lb - 1e-6) and np.all(pm <= ub + 1e-6)

    def test_uniform_sampling(self):
        _x, _lb, _ub, (_pm, w), n_samples = self.run()
        ref = das_dennis_reference(20, 3)
        assert isinstance(n_samples, int)
        assert n_samples == ref.shape[0]
        assert w.shape == (n_samples, 3) and w.dtype == np.float32
        np.testing.assert_allclose(w, ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(w.sum(axis=1), np.ones(n_samples), rtol=1e-5, atol=1e-5)

    def test_determinism(self):
        x, _lb, _ub, (pm1, w1), _ = self.run()
        pm2, w2 = self.run()[3]
        assert np.array_equal(pm1, pm2) and np.array_equal(w1, w2)
