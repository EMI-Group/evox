"""Parity tests: functional evox_etl metrics (etl numpy backend) vs torch evox.
This is the ONLY file in the suite allowed to import torch (DESIGN §6).
"""
import numpy as np
import pytest
import torch
import etl
from etl.core import TensorSpec

import evox.metrics as m_torch

from evox_etl.metrics import gd, igd

F32 = np.float32


def _as_np(x):
    return x.numpy() if hasattr(x, "numpy") else x


@pytest.mark.parametrize("shapes,seed", [((7, 10, 3), 0), ((1, 5, 2), 1), ((20, 30, 4), 2)])
def test_gd_parity_with_torch(shapes, seed):
    n, k, m = shapes
    rng = np.random.default_rng(seed)
    objs = rng.uniform(0, 2, (n, m)).astype(F32)
    pf = rng.uniform(0, 2, (k, m)).astype(F32)
    expected = m_torch.gd(torch.from_numpy(objs), torch.from_numpy(pf)).item()
    exe = etl.build(gd, TensorSpec((n, m), F32), TensorSpec((k, m), F32), backend="numpy")
    got = float(_as_np(etl.run(exe, etl.core.tensor(objs), etl.core.tensor(pf))))
    assert abs(got - expected) <= 1e-4


@pytest.mark.parametrize("shapes,seed", [((7, 10, 3), 0), ((1, 5, 2), 1), ((20, 30, 4), 2)])
@pytest.mark.parametrize("p", [1.0, 2.0])
def test_igd_parity_with_torch(shapes, seed, p):
    n, k, m = shapes
    rng = np.random.default_rng(seed)
    objs = rng.uniform(0, 2, (n, m)).astype(F32)
    pf = rng.uniform(0, 2, (k, m)).astype(F32)
    expected = m_torch.igd(torch.from_numpy(objs), torch.from_numpy(pf), p=p).item()
    exe = etl.build(igd, TensorSpec((n, m), F32), TensorSpec((k, m), F32), p, backend="numpy")
    got = float(_as_np(etl.run(exe, etl.core.tensor(objs), etl.core.tensor(pf), p)))
    assert abs(got - expected) <= 1e-4
