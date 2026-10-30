"""Pure-EvoX tests for the evox metrics (NO torch imports here).

Everything runs through ``evoxc.build`` + ``evoxc.run`` on the default numpy
backend. Host-side contract (verified empirically, see
unit_test/evox/algorithms/helpers.py):
* static Python-scalar args must be passed to ``evoxc.build`` AND re-passed
  (by value) to ``evoxc.run``;
* ``evoxc.run`` returns concrete ``evoxc.core.tensor.Tensor`` objects — use
  ``.numpy()`` to get ndarrays (np.asarray gives object dtype).
"""
import itertools

import evoxc
import evoxc.numpy as enp
import evoxc.random as rnd
import numpy as np
import pytest
from evoxc.core import TensorSpec

from evox.metrics import (
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
    exe = evoxc.build(fn, *specs, backend="numpy")
    tensors = [evoxc.core.tensor(a) if isinstance(a, np.ndarray) else a for a in args]
    return evoxc.run(exe, *tensors)


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
    points = evoxc.abs(objs - ref)
    bound = evoxc.max(points, axes=0)
    max_vol = evoxc.prod(bound, axes=None)
    samples = rnd.uniform(key, (num_sample, points.shape[1]), 0.0, bound)
    dominated = enp.expand_dims(samples, 1) < enp.expand_dims(points, 0)
    in_any = evoxc.max(evoxc.min(dominated, axes=-1), axes=-1)
    count = evoxc.sum(in_any, axes=None)
    return evoxc.cast(count, 'float32') / float(num_sample) * max_vol, samples


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
        points = evoxc.abs(objs_ - ref_)
        n = points.shape[0]
        k = num_sample // n
        keys = rnd.split_n(key, n)
        total = 0.0
        samples_list = []
        for i in range(n):
            samples_i = rnd.uniform(keys[i], (k, points.shape[1]), 0.0, enp.expand_dims(points[i], 0))
            samples_list.append(samples_i)
            dominated = enp.expand_dims(samples_i, 1) < enp.expand_dims(points, 0)
            dom_count = evoxc.sum(evoxc.min(dominated, axes=-1), axes=-1)
            share = evoxc.select(dom_count > 0, 1.0 / evoxc.maximum(evoxc.cast(dom_count, 'float32'), 1.0), 0.0)
            total = total + evoxc.prod(points[i], axes=None) * evoxc.sum(share, axes=None) / float(k)
        return evoxc.cast(total, 'float32'), samples_list

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


# --- non-dominating reference-point clipping (hv / each_cube_monte_carlo_hv) --
# A point that is NOT better than the reference point along some objective has
# no extent along that objective: objectives are minimized, so the box a point
# covers towards ``ref`` is ``max(ref - objs, 0)``. A regression to
# ``abs(ref - objs)`` (or to an unclipped ``ref - objs``) turns such a point
# into a contributor and inflates every estimate below.


def _union_volume(points):
    """Exact volume of the union of origin-anchored boxes [0, p] (inclusion-exclusion)."""
    total = 0.0
    n = points.shape[0]
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            total += (-1.0) ** (r + 1) * float(points[list(combo)].min(axis=0).prod())
    return total


def _hv_value(fn, seed, objs, ref, num_sample):
    specs = [
        TensorSpec((), np.int64),
        TensorSpec(objs.shape, F32),
        TensorSpec(ref.shape, F32),
        num_sample,
    ]
    return float(_as_np(_run(fn, rnd.key(seed).numpy(), objs, ref, num_sample, specs=specs)))


HV_FNS = [hv, each_cube_monte_carlo_hv]
HV_FN_IDS = ["bounding_cube", "each_cube"]


@pytest.mark.parametrize("fn", HV_FNS, ids=HV_FN_IDS)
def test_hv_point_outside_ref_contributes_nothing(fn):
    objs = np.array([[1.5, 0.5]], dtype=F32)  # worse than ref along objective 0
    ref = np.array([1.0, 1.0], dtype=F32)
    assert _hv_value(fn, 0, objs, ref, 100000) == 0.0


@pytest.mark.parametrize("fn", HV_FNS, ids=HV_FN_IDS)
def test_hv_half_outside_front_matches_dominated_area(fn):
    objs = np.array([[0.2, 0.6], [1.5, 0.5]], dtype=F32)
    ref = np.array([1.0, 1.0], dtype=F32)
    # clipped boxes: [0.8, 0.4] (area 0.32) and [0, 0.5] (area 0) → 0.32
    got = _hv_value(fn, 0, objs, ref, 400000)
    assert got == pytest.approx(0.32, abs=0.01)


@pytest.mark.parametrize("fn", HV_FNS, ids=HV_FN_IDS)
def test_hv_three_objective_mixed_front_ignores_outside_point(fn):
    ref = np.ones(3, dtype=F32)
    front = np.array(
        [[0.2, 0.3, 0.4], [0.5, 0.1, 0.2], [0.1, 0.6, 0.3]], dtype=F32
    )
    outside = np.array([[0.5, 0.5, 3.0]], dtype=F32)  # violates objective 2 only
    expected = _union_volume(np.maximum(ref - front, 0.0))
    got = _hv_value(fn, 0, front, ref, 200000)
    assert got == pytest.approx(expected, abs=0.02)
    # per-dimension clipping: the outside point has zero extent along the
    # violated objective, so it must not add hypervolume (an ``abs`` regression
    # inflates the estimate from ~0.53 to ~0.83 here)
    with_outside = _hv_value(fn, 0, np.vstack([front, outside]), ref, 200000)
    assert with_outside == pytest.approx(got, abs=0.01)
    # sanity: without clipping that extra point WOULD contribute (guards the
    # test itself against a no-op front)
    unclipped = _union_volume(np.abs(np.vstack([front, outside]) - ref))
    assert unclipped > expected + 0.1
