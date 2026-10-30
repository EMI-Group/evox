"""Pure-ETL tests for the CEC2022 suite (NO torch imports here).

Covers running every valid (function, dimension) pair on a random population,
the config assertions inherited from the reference (invalid dimension, f6-8 at
D=2),
error paths, and the f1 shift-point check (fitness == 300 at the optimum).
Everything runs through ``evoxc.build`` + ``evoxc.run`` on the default numpy
backend.
"""

import evoxc
import evoxc.core
import numpy as np
import pytest

from evox_etl.problems.numerical import cec2022 as cec_module
from evox_etl.problems.numerical.cec2022 import CEC2022, evaluate
from evox_etl.problems.numerical.state import ProblemState

# The port ships the CEC2022 data inside its own package, so reuse the module's
# own resolution (a Traversable) instead of reaching into another source tree.
DATA_DIR = cec_module.DATA_DIR

N = 8

# every valid (function, dimension) combination: f6-8 are only defined for
# D=10 and D=20
ALL_CASES = [
    (func_num, dim)
    for func_num in range(1, 13)
    for dim in (2, 10, 20)
    if not (func_num in (6, 7, 8) and dim == 2)
]


def run_cec(func_num, dim, x):
    """Trace ``evaluate(CEC2022(func_num, dim), ProblemState(), pop)`` and run it."""
    config = CEC2022(func_num, dim)
    spec = evoxc.core.TensorSpec(x.shape, np.dtype("float32"))
    exe = evoxc.build(evaluate, config, ProblemState(), spec)
    fitness, problem_state = evoxc.run(exe, config, ProblemState(), x)
    return fitness.numpy(), problem_state


def random_pop(rng, dim, n=N):
    return rng.uniform(-100.0, 100.0, (n, dim)).astype(np.float32)


# --- basic run --------------------------------------------------------------


def test_f1_basic_run():
    x = random_pop(np.random.default_rng(0), 10)
    fitness, problem_state = run_cec(1, 10, x)
    assert fitness.shape == (N,)
    assert fitness.dtype == np.float32
    assert np.all(np.isfinite(fitness))
    assert problem_state == ProblemState()


# --- all valid functions run and stay finite --------------------------------


@pytest.mark.parametrize("func_num, dim", ALL_CASES)
def test_all_functions_finite(func_num, dim):
    x = random_pop(np.random.default_rng(func_num * 100 + dim), dim)
    fitness, _ = run_cec(func_num, dim, x)
    assert np.all(np.isfinite(fitness))


# --- config validation (mirrors the CEC2022 config contract) -----------------


def test_invalid_dimension_rejected():
    with pytest.raises(AssertionError, match="only defined for D=2,10,20"):
        CEC2022(1, 5)


def test_f6_not_defined_for_d2():
    with pytest.raises(AssertionError, match="not defined for D=2"):
        CEC2022(6, 2)


def test_unknown_function_number_rejected():
    # _load_data runs (at trace time) before the dispatch, so a missing data
    # file surfaces first — the reference loads its data in __init__ too.
    with pytest.raises(FileNotFoundError, match="Cannot open"):
        run_cec(13, 10, np.zeros((2, 10), dtype=np.float32))


def test_pop_dimension_mismatch_rejected():
    with pytest.raises(AssertionError, match="Dimension mismatch"):
        run_cec(1, 10, np.zeros((2, 5), dtype=np.float32))


# --- f1 shift point ----------------------------------------------------------


def test_f1_shift_point_is_minimum():
    # at pop == OShift (first `dim` entries of the shift data file), f1 is
    # zakharov(0) + bias = 300
    shift_row = np.array(
        [float(v) for v in (DATA_DIR / "shift_data_1.txt").read_text().split()],
        dtype=np.float32,
    )
    pop = shift_row[:10].reshape(1, 10)
    fitness, _ = run_cec(1, 10, pop)
    np.testing.assert_allclose(fitness, [300.0], atol=10.0)
