"""Parity tests: evox_etl jit-fix operator utils vs the torch evox reference.

The four helpers covered here (``switch``, ``clip``, ``maximum_float``,
``minimum_float``) are deterministic (no key), so exact parity at 1e-6 is
required — the etl ports are literal translations of the torch
``src/evox/utils/jit_fix_operator.py`` relu/where compositions. Both sides run
on the SAME numpy inputs: the etl side via ``etl.build`` + ``etl.run`` (numpy
backend; etl has no eager mode, the helpers are trace-only), the torch side via
``torch.tensor`` inputs. torch imports are allowed only in this ``parity/``
subpackage.
"""

import etl
import numpy as np
import pytest
import torch
from etl import core

from evox.utils.jit_fix_operator import (
    clip as torch_clip,
)
from evox.utils.jit_fix_operator import (
    maximum_float as torch_maximum_float,
)
from evox.utils.jit_fix_operator import (
    minimum_float as torch_minimum_float,
)
from evox.utils.jit_fix_operator import (
    switch as torch_switch,
)
from evox_etl.operators.jit_fix_operator import (
    clip,
    maximum_float,
    minimum_float,
    switch,
)

F32 = np.dtype("float32")
I32 = np.dtype("int32")

LABEL_SPEC = core.TensorSpec((6,), I32)
VEC_SPEC = core.TensorSpec((6,), F32)

LABELS = np.array([0, 1, 2, 2, 1, 0], dtype=np.int32)
SWITCH_VALUES = [
    np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
    np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], dtype=np.float32),
    np.array([0.5, 0.25, 0.125, 0.0625, 0.125, 0.25], dtype=np.float32),
]

# values on both sides of [0, 1] plus exact boundary hits
CLIP_IN = np.array([-0.5, 0.0, 0.3, 1.0, 2.0, -1e-7], dtype=np.float32)

MAXMIN_IN = np.array([-2.0, -0.5, 0.0, 0.5, 1.0, 7.0], dtype=np.float32)


def _build(fn, *specs):
    return etl.build(fn, *specs, backend="numpy")


def test_switch_parity_three_values():
    exe = _build(switch, LABEL_SPEC, [VEC_SPEC] * 3)
    out = etl.run(exe, LABELS, SWITCH_VALUES).numpy()
    t_out = torch_switch(torch.tensor(LABELS), [torch.tensor(v) for v in SWITCH_VALUES]).numpy()
    assert out.dtype == t_out.dtype == np.float32
    np.testing.assert_allclose(out, t_out, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n_values", [2, 4], ids=["two-values", "four-values"])
def test_switch_parity_varied_arity(n_values):
    values = [np.full(6, float(k) - 1.0, dtype=np.float32) for k in range(n_values)]
    exe = _build(switch, LABEL_SPEC, [VEC_SPEC] * n_values)
    out = etl.run(exe, LABELS, values).numpy()
    t_out = torch_switch(torch.tensor(LABELS), [torch.tensor(v) for v in values]).numpy()
    np.testing.assert_allclose(out, t_out, rtol=1e-6, atol=1e-6)


def test_clip_parity():
    exe = _build(clip, VEC_SPEC)
    out = etl.run(exe, CLIP_IN).numpy()
    t_out = torch_clip(torch.tensor(CLIP_IN)).numpy()
    assert out.dtype == t_out.dtype == np.float32
    np.testing.assert_allclose(out, t_out, rtol=1e-6, atol=1e-6)
    # sanity: the torch reference really does map into [0, 1]
    np.testing.assert_allclose(t_out, np.array([0.0, 0.0, 0.3, 1.0, 1.0, 0.0], dtype=np.float32))


@pytest.mark.parametrize("b", [-1.0, 0.0, 0.5, 3.0], ids=lambda b: f"b={b}")
def test_maximum_float_parity(b):
    exe = _build(maximum_float, VEC_SPEC, float(b))
    out = etl.run(exe, MAXMIN_IN, float(b)).numpy()
    t_out = torch_maximum_float(torch.tensor(MAXMIN_IN), float(b)).numpy()
    assert out.dtype == t_out.dtype == np.float32
    np.testing.assert_allclose(out, t_out, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("b", [-1.0, 0.0, 0.5, 3.0], ids=lambda b: f"b={b}")
def test_minimum_float_parity(b):
    exe = _build(minimum_float, VEC_SPEC, float(b))
    out = etl.run(exe, MAXMIN_IN, float(b)).numpy()
    t_out = torch_minimum_float(torch.tensor(MAXMIN_IN), float(b)).numpy()
    assert out.dtype == t_out.dtype == np.float32
    np.testing.assert_allclose(out, t_out, rtol=1e-6, atol=1e-6)
