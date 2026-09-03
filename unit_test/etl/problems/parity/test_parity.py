"""Parity tests: functional evox_etl (etl numpy backend) vs torch evox.

This is the ONLY file in the suite allowed to import torch (DESIGN §6).  The
same seeded numpy populations are evaluated by both implementations and
compared.  Tolerances: basic problems are exact to float32 rounding (1e-5);
DTLZ evaluate/pf within 1e-4 (the DTLZ5/6 pf involve a descending float32
arange whose torch/numpy kernels drift apart by ~5e-5); CEC2022 magnitudes
reach ~1e10, so a relative 1e-4 tolerance is used there.
"""

import numpy as np
import pytest
import torch

import etl
import etl.core

import evox.problems.numerical as torch_num
import evox.problems.numerical.basic as torch_basic  # Zakharov/Levy live here only

import evox_etl.problems.numerical as etl_num
import evox_etl.problems.numerical.basic as etl_basic
import evox_etl.problems.numerical.dtlz as etl_dtlz
from evox_etl.problems.numerical.state import ProblemState

RNG = np.random.default_rng(2024)

# (name, etl config class, torch problem class, default d)
BASIC_CASES = [
    ("Ackley", etl_basic.Ackley, torch_num.Ackley),
    ("Griewank", etl_basic.Griewank, torch_num.Griewank),
    ("Rastrigin", etl_basic.Rastrigin, torch_num.Rastrigin),
    ("Rosenbrock", etl_basic.Rosenbrock, torch_num.Rosenbrock),
    ("Schwefel", etl_basic.Schwefel, torch_num.Schwefel),
    ("Sphere", etl_basic.Sphere, torch_num.Sphere),
    ("Ellipsoid", etl_basic.Ellipsoid, torch_num.Ellipsoid),
    ("Zakharov", etl_basic.Zakharov, torch_basic.Zakharov),
    ("Levy", etl_basic.Levy, torch_basic.Levy),
]

# (name, etl config class, torch problem class, default d)
DTLZ_CASES = [
    ("DTLZ1", etl_dtlz.DTLZ1, torch_num.DTLZ1, 7),
    ("DTLZ2", etl_dtlz.DTLZ2, torch_num.DTLZ2, 12),
    ("DTLZ3", etl_dtlz.DTLZ3, torch_num.DTLZ3, 12),
    ("DTLZ4", etl_dtlz.DTLZ4, torch_num.DTLZ4, 12),
    ("DTLZ5", etl_dtlz.DTLZ5, torch_num.DTLZ5, 12),
    ("DTLZ6", etl_dtlz.DTLZ6, torch_num.DTLZ6, 12),
    ("DTLZ7", etl_dtlz.DTLZ7, torch_num.DTLZ7, 21),
]

# CEC2022: all 12 functions at dim 10, plus f1/f9 at dims 2 and 20
CEC_CASES = [(f, 10) for f in range(1, 13)] + [(1, 2), (9, 2), (1, 20), (9, 20)]


def run_etl(fn, config, x):
    """Trace ``fn(config, ProblemState(), pop)`` and run it on ``x`` (numpy)."""
    spec = etl.core.TensorSpec(x.shape, np.dtype("float32"))
    exe = etl.build(fn, config, ProblemState(), spec)
    fitness, _ = etl.run(exe, config, ProblemState(), x)
    return fitness.numpy()


def run_etl_pf(fn, config):
    """Trace ``fn(config)`` (no tensor args) and run it."""
    exe = etl.build(fn, config)
    return etl.run(exe, config).numpy()


def run_torch(torch_cls, x, **kwargs):
    """Evaluate a torch evox problem class on the (numpy) population."""
    return torch_cls(**kwargs).evaluate(torch.tensor(x)).numpy()


# --- basic problems ---------------------------------------------------------


@pytest.mark.parametrize("name, etl_cls, torch_cls", BASIC_CASES)
def test_basic_default_parity(name, etl_cls, torch_cls):
    x = RNG.standard_normal((16, 10)).astype(np.float32)
    f_etl = run_etl(etl_basic.evaluate, etl_cls(), x)
    f_torch = run_torch(torch_cls, x)
    np.testing.assert_allclose(f_etl, f_torch, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name, etl_cls, torch_cls", BASIC_CASES)
def test_basic_shift_affine_parity(name, etl_cls, torch_cls):
    shift = RNG.standard_normal(10).astype(np.float32)
    affine = RNG.standard_normal((10, 10)).astype(np.float32)
    x = RNG.standard_normal((16, 10)).astype(np.float32)
    f_etl = run_etl(etl_basic.evaluate, etl_cls(shift=shift, affine=affine), x)
    f_torch = run_torch(
        torch_cls, x, shift=torch.tensor(shift), affine=torch.tensor(affine)
    )
    np.testing.assert_allclose(f_etl, f_torch, rtol=1e-4, atol=1e-4)


# --- DTLZ --------------------------------------------------------------------


@pytest.mark.parametrize("name, etl_cls, torch_cls, d", DTLZ_CASES)
def test_dtlz_evaluate_parity(name, etl_cls, torch_cls, d):
    x = RNG.random((64, d)).astype(np.float32)
    f_etl = run_etl(etl_dtlz.evaluate, etl_cls(), x)
    f_torch = run_torch(torch_cls, x)
    np.testing.assert_allclose(f_etl, f_torch, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name, etl_cls, torch_cls, d", DTLZ_CASES)
def test_dtlz_pf_parity(name, etl_cls, torch_cls, d):
    pf_etl = run_etl_pf(etl_dtlz.pf, etl_cls())
    pf_torch = torch_cls().pf().numpy()
    # 1e-4: DTLZ5/6 pf are built from a descending float32 arange
    # (torch.arange(1, 0, -1/(n-1))); torch's and numpy's float32 arange
    # kernels accumulate the step differently and drift apart by up to ~5e-5
    # (ascending arange matches to 1 ulp).  The formulas are 1:1 — this is a
    # backend rounding divergence, not a math difference.
    np.testing.assert_allclose(pf_etl, pf_torch, rtol=1e-4, atol=1e-4)


# --- CEC2022 -----------------------------------------------------------------


@pytest.mark.parametrize("func_num, dim", CEC_CASES)
def test_cec2022_parity(func_num, dim):
    x = RNG.uniform(-100.0, 100.0, (4, dim)).astype(np.float32)
    f_etl = run_etl(
        etl_num.cec2022.evaluate, etl_num.CEC2022(func_num, dim), x
    )
    f_torch = run_torch(
        torch_num.CEC2022, x, problem_number=func_num, dimension=dim, device="cpu"
    )
    np.testing.assert_allclose(f_etl, f_torch, rtol=1e-4, atol=1e-2)
