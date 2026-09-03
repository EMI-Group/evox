"""Pure-etl tests for the evox_etl metrics (NO torch imports here).

Everything runs through ``etl.build`` + ``etl.run`` on the default numpy
backend. Host-side contract (verified empirically, see
unit_test/etl/algorithms/helpers.py):
* static Python-scalar args must be passed to ``etl.build`` AND re-passed
  (by value) to ``etl.run``;
* ``etl.run`` returns concrete ``etl.core.tensor.Tensor`` objects — use
  ``.numpy()`` to get ndarrays (np.asarray gives object dtype).
"""
import numpy as np
import pytest
import etl
import etl.numpy as enp
import etl.random as rnd
from etl.core import TensorSpec

from evox_etl.metrics import (
    bounding_cube_monte_carlo_hv,
    each_cube_monte_carlo_hv,
    gd,
    gd_plus,
    hv,
    igd,
    igd_plus,
)

F32 = np.float32


def _as_np(x):
    return x.numpy() if hasattr(x, "numpy") else x


def _run(fn, *args, specs):
    exe = etl.build(fn, *specs, backend="numpy")
    tensors = [etl.core.tensor(a) if isinstance(a, np.ndarray) else a for a in args]
    return etl.run(exe, *tensors)


def _gd_np(objs, pf):
    dist = np.sqrt(((objs[:, None, :] - pf[None, :, :]) ** 2).sum(-1))
    return np.linalg.norm(dist.min(axis=1)) / objs.shape[0]


def _igd_np(objs, pf, p):
    dist = np.sqrt(((pf[:, None, :] - objs[None, :, :]) ** 2).sum(-1))
    return (dist.min(axis=1) ** p).mean() ** (1.0 / p)


def _gd_plus_np(objs, pf, p):
    diff = pf[None, :, :] - objs[:, None, :]
    d = np.sqrt((np.maximum(diff, 0.0) ** 2).sum(-1))
    return (d.min(axis=1) ** p).sum() ** (1.0 / p) / objs.shape[0] ** (1.0 / p)


def _igd_plus_np(objs, pf, p):
    diff = objs[None, :, :] - pf[:, None, :]
    d = np.sqrt((np.maximum(diff, 0.0) ** 2).sum(-1))
    return (d.min(axis=1) ** p).sum() ** (1.0 / p) / pf.shape[0] ** (1.0 / p)


CASES = [((7, 10, 3), 0), ((1, 5, 2), 1), ((20, 30, 4), 2), ((13, 7, 5), 3)]


@pytest.mark.parametrize("shapes,seed", CASES)
def test_gd_matches_numpy_reference(shapes, seed):
    n, k, m = shapes
    rng = np.random.default_rng(seed)
    objs = rng.uniform(0, 2, (n, m)).astype(F32)
    pf = rng.uniform(0, 2, (k, m)).astype(F32)
    got = float(_as_np(_run(gd, objs, pf, specs=[TensorSpec((n, m), F32), TensorSpec((k, m), F32)])))
    assert abs(got - _gd_np(objs, pf)) <= 1e-5


@pytest.mark.parametrize("shapes,seed", CASES)
@pytest.mark.parametrize("p", [1.0, 2.0, 4.0])
def test_igd_matches_numpy_reference(shapes, seed, p):
    n, k, m = shapes
    rng = np.random.default_rng(seed)
    objs = rng.uniform(0, 2, (n, m)).astype(F32)
    pf = rng.uniform(0, 2, (k, m)).astype(F32)
    got = float(_as_np(_run(igd, objs, pf, p, specs=[TensorSpec((n, m), F32), TensorSpec((k, m), F32), p])))
    assert abs(got - _igd_np(objs, pf, p)) <= 1e-5


@pytest.mark.parametrize("shapes,seed", CASES)
@pytest.mark.parametrize("p", [1.0, 2.0, 4.0])
def test_gd_plus_matches_numpy_reference(shapes, seed, p):
    n, k, m = shapes
    rng = np.random.default_rng(seed)
    objs = rng.uniform(0, 2, (n, m)).astype(F32)
    pf = rng.uniform(0, 2, (k, m)).astype(F32)
    got = float(_as_np(_run(gd_plus, objs, pf, p, specs=[TensorSpec((n, m), F32), TensorSpec((k, m), F32), p])))
    assert abs(got - _gd_plus_np(objs, pf, p)) <= 1e-5


@pytest.mark.parametrize("shapes,seed", CASES)
@pytest.mark.parametrize("p", [1.0, 2.0, 4.0])
def test_igd_plus_matches_numpy_reference(shapes, seed, p):
    n, k, m = shapes
    rng = np.random.default_rng(seed)
    objs = rng.uniform(0, 2, (n, m)).astype(F32)
    pf = rng.uniform(0, 2, (k, m)).astype(F32)
    got = float(_as_np(_run(igd_plus, objs, pf, p, specs=[TensorSpec((n, m), F32), TensorSpec((k, m), F32), p])))
    assert abs(got - _igd_plus_np(objs, pf, p)) <= 1e-5


def _hv_with_samples(key, objs, ref, num_sample):
    points = etl.abs(objs - ref)
    bound = etl.max(points, axes=0)
    max_vol = etl.prod(bound, axes=None)
    samples = rnd.uniform(key, (num_sample, points.shape[1]), 0.0, bound)
    dominated = enp.expand_dims(samples, 1) < enp.expand_dims(points, 0)
    in_any = etl.max(etl.min(dominated, axes=-1), axes=-1)
    count = etl.sum(in_any, axes=None)
    return etl.cast(count, 'float32') / float(num_sample) * max_vol, samples


@pytest.mark.parametrize("seed,num_sample", [(0, 20000), (7, 5000)])
def test_hv_matches_recomputation_from_drawn_samples(seed, num_sample):
    rng = np.random.default_rng(seed)
    objs = rng.uniform(0, 2, (6, 3)).astype(F32)
    ref = np.array([3.0, 3.0, 3.0], dtype=F32)
    key_np = rnd.key(seed).numpy()
    specs = [TensorSpec((), np.int64), TensorSpec(objs.shape, F32), TensorSpec(ref.shape, F32), num_sample]

    got, samples = _run(_hv_with_samples, key_np, objs, ref, num_sample, specs=specs)
    got = float(_as_np(got))
    samples_np = _as_np(samples)
    points_np = np.abs(objs - ref)
    bound_np = points_np.max(axis=0)
    max_vol_np = bound_np.prod()
    in_np = np.any(np.all(samples_np[:, None, :] < points_np[None, :, :], axis=2), axis=1).sum()
    expected = np.float32(in_np) / np.float32(num_sample) * np.float32(max_vol_np)
    assert abs(got - float(expected)) <= 1e-6
    assert 0.0 <= got <= max_vol_np + 1e-6

    for alias in (hv, bounding_cube_monte_carlo_hv):
        alias_val = float(_as_np(_run(alias, key_np, objs, ref, num_sample, specs=specs)))
        assert abs(alias_val - got) <= 1e-6


@pytest.mark.parametrize("seed", [3, 5])
def test_each_cube_hv_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    objs = rng.uniform(0, 2, (3, 2)).astype(F32)
    ref = np.array([3.0, 3.0], dtype=F32)
    num_sample = 3000
    key_np = rnd.key(seed).numpy()

    def each_cube_with_samples(key, objs_, ref_):
        points = etl.abs(objs_ - ref_)
        n = points.shape[0]
        k = num_sample // n
        keys = rnd.split_n(key, n)
        total = 0.0
        samples_list = []
        for i in range(n):
            samples_i = rnd.uniform(keys[i], (k, points.shape[1]), 0.0, enp.expand_dims(points[i], 0))
            samples_list.append(samples_i)
            dominated = enp.expand_dims(samples_i, 1) < enp.expand_dims(points, 0)
            dom_count = etl.sum(etl.min(dominated, axes=-1), axes=-1)
            share = etl.select(dom_count > 0, 1.0 / etl.maximum(etl.cast(dom_count, 'float32'), 1.0), 0.0)
            total = total + etl.prod(points[i], axes=None) * etl.sum(share, axes=None) / float(k)
        return etl.cast(total, 'float32'), samples_list

    specs = [TensorSpec((), np.int64), TensorSpec(objs.shape, F32), TensorSpec(ref.shape, F32)]
    got, samples_list = _run(each_cube_with_samples, key_np, objs, ref, specs=specs)
    got = float(_as_np(got))

    points_np = np.abs(objs - ref)
    n = objs.shape[0]
    k = num_sample // n
    total_np = 0.0
    for i in range(n):
        samples_i = _as_np(samples_list[i])
        assert samples_i.shape == (k, 2)
        dom = (samples_i[:, None, :] < points_np[None, :, :]).all(-1).sum(-1)
        share = np.where(dom > 0, 1.0 / np.maximum(dom.astype(F32), 1.0), 0.0)
        total_np += points_np[i].prod() * share.sum() / k
    assert abs(got - total_np) <= 1e-5

    public_val = float(_as_np(_run(each_cube_monte_carlo_hv, key_np, objs, ref, num_sample, specs=specs + [num_sample])))
    assert abs(public_val - got) <= 1e-6
