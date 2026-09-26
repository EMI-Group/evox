"""Parity tests: evox_etl crossover operators vs the torch evox reference (torch allowed here).

``DE_arithmetic_recombination`` is deterministic (no key), so exact parity at
1e-6 is required. Both sides run on the SAME numpy inputs: the etl side via
``etl.build`` + ``etl.run`` (numpy backend; etl has no eager mode), the torch
side via ``torch.tensor`` inputs. The three supported K shapes are covered:
python-float scalar (static at build/run), (pop,) vector and (pop, 1) matrix.
"""

import numpy as np
import pytest
import torch

import etl
from etl import core

from evox.operators.crossover import DE_arithmetic_recombination as torch_DE_arithmetic_recombination
from evox_etl.operators.crossover import DE_arithmetic_recombination

RNG = np.random.default_rng(0)

F32 = np.dtype("float32")
POP_SPEC = core.TensorSpec((32, 10), F32)

MV = RNG.standard_normal((32, 10)).astype(np.float32)
CV = RNG.standard_normal((32, 10)).astype(np.float32)


def test_de_arithmetic_recombination_parity_scalar_k():
    # K = 0.5 as a python-float static, re-passed at run (issue 17)
    exe = etl.build(DE_arithmetic_recombination, POP_SPEC, POP_SPEC, 0.5, backend="numpy")
    out = etl.run(exe, MV, CV, 0.5).numpy()
    t_out = torch_DE_arithmetic_recombination(
        torch.tensor(MV), torch.tensor(CV), torch.tensor(0.5)
    ).numpy()
    assert out.dtype == t_out.dtype == np.float32
    np.testing.assert_allclose(out, t_out, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(32,), (32, 1)], ids=["vector", "matrix"])
def test_de_arithmetic_recombination_parity_tensor_k(shape):
    K = RNG.random(shape).astype(np.float32)
    exe = etl.build(DE_arithmetic_recombination, POP_SPEC, POP_SPEC,
                    core.TensorSpec(shape, F32), backend="numpy")
    out = etl.run(exe, MV, CV, K).numpy()
    t_out = torch_DE_arithmetic_recombination(
        torch.tensor(MV), torch.tensor(CV), torch.tensor(K)
    ).numpy()
    np.testing.assert_allclose(out, t_out, rtol=1e-6, atol=1e-6)
