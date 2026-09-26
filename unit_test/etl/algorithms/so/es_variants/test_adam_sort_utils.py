"""Smoke tests for the es_variants shared helpers (adam_step, sort_utils).

ETL has no eager mode: the helpers are plain functions exercised inside a
traced wrapper via `etl.build`/`etl.run` (numpy backend). No torch imports.
"""

import numpy as np

import etl
from etl import core

from evox_etl.algorithms.so.es_variants.adam_step import adam_single_tensor
from evox_etl.algorithms.so.es_variants.sort_utils import sort_by_key


def _spec(a):
    return core.TensorSpec(shape=tuple(a.shape), dtype=np.dtype(a.dtype))


def test_adam_single_tensor():
    rng = np.random.default_rng(42)
    param = rng.standard_normal((5,)).astype(np.float32)
    grad = rng.standard_normal((5,)).astype(np.float32)
    exp_avg = rng.standard_normal((5,)).astype(np.float32)
    exp_avg_sq = np.abs(rng.standard_normal((5,)).astype(np.float32))

    beta1, beta2, lr, wd, eps = 0.9, 0.999, 1e-3, 1e-2, 1e-8

    def _apply(p, g, ea, eas):
        return adam_single_tensor(
            p, g, ea, eas, beta1=beta1, beta2=beta2, lr=lr,
            weight_decay=wd, eps=eps,
        )

    exe = etl.build(
        _apply, _spec(param), _spec(grad), _spec(exp_avg), _spec(exp_avg_sq),
        backend="numpy",
    )
    new_param, new_ea, new_eas = [
        t.numpy() for t in etl.run(exe, param, grad, exp_avg, exp_avg_sq)
    ]

    g = grad + wd * param
    ea = beta1 * exp_avg + (1.0 - beta1) * g
    eas = exp_avg_sq * beta2 + g * g * (1.0 - beta2)
    denom = np.sqrt(eas) + eps
    np.testing.assert_allclose(new_param, param - lr * ea / denom, rtol=0, atol=1e-4)
    np.testing.assert_allclose(new_ea, ea, rtol=0, atol=1e-4)
    np.testing.assert_allclose(new_eas, eas, rtol=0, atol=1e-4)


def test_sort_by_key():
    rng = np.random.default_rng(7)
    keys = rng.standard_normal((10,)).astype(np.float32)
    pop = rng.standard_normal((10, 3)).astype(np.float32)

    exe = etl.build(
        sort_by_key, _spec(keys), _spec(pop), backend="numpy"
    )
    sk, sp = [t.numpy() for t in etl.run(exe, keys, pop)]

    np.testing.assert_allclose(sk, np.sort(keys), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(sp, pop[np.argsort(keys)])
