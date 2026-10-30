"""Regression tests for the initial population of CoDE / SaDE / SHADE (no torch).

The initial population must be drawn UNIFORMLY in ``[lb, ub]`` (upstream fix
ported into ``evox_etl.algorithms.so.de_variants.{code,sade,shade}``). The
test uses a tight box ``[3, 4]``: a uniform draw keeps every gene inside and
covers the box roughly uniformly (mean 3.5, std 1/sqrt(12) ~ 0.2887), while a
normal draw around the box visibly escapes it.
"""
import sys
from pathlib import Path

sys.path[0:0] = [
    str(Path(__file__).resolve().parents[5]),
    str(Path(__file__).resolve().parents[5] / "src"),
]

import evoxc
import numpy as np
import pytest

import evox_etl.algorithms.so.de_variants.code as code_mod
import evox_etl.algorithms.so.de_variants.sade as sade_mod
import evox_etl.algorithms.so.de_variants.shade as shade_mod

POP, DIM = 200, 20
LB, UB = 3.0, 4.0
# uniform-in-[LB, UB] reference moments
UNIFORM_MEAN = 0.5 * (LB + UB)
UNIFORM_STD = (UB - LB) / np.sqrt(12.0)

DE_VARIANTS = [
    pytest.param(code_mod, code_mod.make_code, id="code"),
    pytest.param(sade_mod, sade_mod.make_sade, id="sade"),
    pytest.param(shade_mod, shade_mod.make_shade, id="shade"),
]


def _initial_pop(mod, make, seed=0):
    """Run the algorithm's ``init`` on the tight box and return the population."""
    cfg = make(pop_size=POP, lb=np.full(DIM, LB), ub=np.full(DIM, UB))
    exe = evoxc.build(
        mod.init,
        cfg,
        evoxc.core.TensorSpec(shape=(), dtype=np.dtype("int64")),
        backend="numpy",
    )
    state = evoxc.run(exe, cfg, np.asarray(seed, dtype=np.int64))
    return state.pop.numpy()


@pytest.mark.parametrize("mod,make", DE_VARIANTS)
@pytest.mark.parametrize("seed", [0, 1])
def test_initial_population_is_uniform_in_bounds(mod, make, seed):
    pop = _initial_pop(mod, make, seed=seed)
    assert pop.shape == (POP, DIM) and pop.dtype == np.float32
    # every gene inside the box — a normal draw around it would escape
    assert np.all(pop >= LB) and np.all(pop <= UB)
    # and the box is covered roughly uniformly, not squeezed near its center
    assert pop.mean() == pytest.approx(UNIFORM_MEAN, abs=0.02)
    assert pop.std() == pytest.approx(UNIFORM_STD, abs=0.02)
    col_min, col_max = pop.min(axis=0), pop.max(axis=0)
    assert np.all(col_min < LB + 0.05)
    assert np.all(col_max > UB - 0.05)


@pytest.mark.parametrize("mod,make", DE_VARIANTS)
def test_initial_population_differs_per_seed(mod, make):
    a = _initial_pop(mod, make, seed=0)
    b = _initial_pop(mod, make, seed=1)
    assert not np.array_equal(a, b)
