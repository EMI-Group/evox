"""StdWorkflow regression tests for monitor-batch shape drift.

Algorithms may hand the monitor candidate batches whose leading dim differs
from ``pop_size`` (CoDE evaluates ``3 * pop_size`` trial vectors per
generation; CSO evaluates ``pop_size // 2`` after its init_step). Torch
``EvalMonitor`` stores the full batch as-is, and these tests pin the same
behavior end-to-end through ``StdWorkflow``: no ShapeError at generation 2
(the pre-fix crash), sane convergence on Sphere, full-batch history entries
and a monotone running elite.

These tests go through ``evox_etl.workflows.StdWorkflow`` (the etl workflow —
NOT torch). Numpy backend only; no torch imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from evox_etl.algorithms import make_code, make_cso, make_pso
from evox_etl.problems.numerical.basic import Sphere
from evox_etl.workflows import EvalMonitorConfig, StdWorkflow

LB, UB = [-5.0] * 4, [5.0] * 4

# The four end-to-end drift tests below pin the workflow-level half of the
# policy: StdWorkflow keys its step-exe cache by (resolved variant, state
# signature) and lazily re-traces when tensor-leaf shapes/dtypes drift
# (evox_etl/core/workflow.py `_state_signature`), so algorithms may evaluate
# candidate batches whose leading dim differs from ``pop_size`` — CoDE hands
# the monitor (3n, dim) batches, CSO (n/2, dim) after its init_step — while
# EvalMonitor stores each batch as-is. The monitor semantics themselves are
# pinned torch-identically by test_eval_monitor.py.


def test_code_runs_with_wide_monitor_batches():
    """CoDE's (3*pop_size, dim) evaluate batches must not crash the workflow
    (the pre-fix failure: ShapeError at generation 2)."""
    wf = StdWorkflow(
        make_code(pop_size=12, lb=LB, ub=UB),
        Sphere(),
        monitor=EvalMonitorConfig(),
        opt_direction="min",
        num_generations=6,
    )
    wf.run()  # raises ShapeError before the fix
    mon = wf.monitor
    assert np.isfinite(mon.get_best_fitness())
    assert mon.get_best_fitness() >= 0.0  # Sphere is non-negative


def test_code_converges_and_records_full_batches():
    pop = 12
    wf = StdWorkflow(
        make_code(pop_size=pop, lb=LB, ub=UB),
        Sphere(),
        monitor=EvalMonitorConfig(full_sol_history=True),
        opt_direction="min",
        num_generations=15,
    )
    best = wf.fit()
    mon = wf.monitor
    assert best < 1.0  # CoDE converges on 4-D Sphere in 15 generations
    # torch parity: every generation's history entry is the FULL 3n batch
    assert all(f.shape == (3 * pop,) for f in mon.get_fitness_history())
    assert all(s.shape == (3 * pop, 4) for s in mon.solution_history)
    # the elite pools the whole 3n batch, so the top-k IS the best so far
    assert mon.get_topk_fitness()[0] == pytest.approx(best)
    assert mon.get_best_solution().shape == (4,)


def test_cso_runs_with_narrow_monitor_batches():
    """CSO evaluates pop_size // 2 candidates per generation after its
    init_step — the monitor batches shrink from (pop,) to (pop//2,)."""
    pop = 16
    wf = StdWorkflow(
        make_cso(pop_size=pop, lb=LB, ub=UB),
        Sphere(),
        monitor=EvalMonitorConfig(),
        opt_direction="min",
        num_generations=5,
    )
    best = wf.fit()
    hist = wf.monitor.get_fitness_history()
    assert len(hist) == 5
    assert hist[0].shape == (pop,)  # init_step evaluates the full population
    assert all(f.shape == (pop // 2,) for f in hist[1:])
    assert np.isfinite(best) and best >= 0.0


def test_cso_max_direction_un_negates():
    wf = StdWorkflow(
        make_cso(pop_size=16, lb=LB, ub=UB),
        Sphere(),
        monitor=EvalMonitorConfig(),
        opt_direction="max",
        num_generations=4,
    )
    wf.run()
    best = wf.monitor.get_best_fitness()
    # maximizing Sphere on [-5, 5]^4 pushes towards a corner (100); the
    # accessor un-negates, so the reported best must be positive
    assert best > 0.0


def test_pso_monitor_unchanged_by_pop_size_batches():
    """pop_size-sized batches (no drift) keep the pre-existing behavior."""
    wf = StdWorkflow(
        make_pso(pop_size=20, lb=LB, ub=UB),
        Sphere(),
        monitor=EvalMonitorConfig(),
        opt_direction="min",
        num_generations=30,
    )
    best = wf.fit()
    hist = wf.monitor.get_fitness_history()
    assert len(hist) == 30
    assert all(f.shape == (20,) for f in hist)
    assert best < 1.0
