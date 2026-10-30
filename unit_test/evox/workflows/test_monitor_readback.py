"""StdWorkflow regression tests for the pipelined monitor readback.

``StdWorkflow._readback`` does NOT copy a device-resident monitor leaf right
away: it schedules an ``evoxc.io.sink`` download, parks the
``(target list, handle)`` pair in ``self._pending_readbacks`` and appends the
result at the start of the next generation (``_drain_readbacks``), so the
device→host latency overlaps the next step graph. ``run``/``fit`` drain the
last generation before returning, and the public ``flush()`` exposes the same
drain to callers that drive ``step``/``init_step``/``final_step`` directly.

Host-resident leaves — the numpy backend used throughout this suite — still
take the eager copy path, so these tests pin the pipeline from both sides:

* deferred-vs-eager history equality on a real PSO + Sphere run, plus the
  empty ``_pending_readbacks`` invariant after ``run()``;
* the deferral itself, via a fake device leaf (``device.kind == "cuda"``) and a
  fake ``evoxc.io.sink`` handle — no GPU is touched;
* ``flush()`` idempotence (calling it twice appends nothing).

Numpy backend only; no torch imports, no cuda/rocm device work.
"""

from __future__ import annotations

import evoxc
import evoxc.core
import evoxc.io
import numpy as np

from evox.algorithms import make_pso
from evox.problems.numerical.basic import Sphere
from evox.workflows import EvalMonitorConfig, StdWorkflow

_DIM = 4
_POP_SIZE = 10
_LB, _UB = [-5.0] * _DIM, [5.0] * _DIM
_CPU = evoxc.core.Device("cpu")


def _make_workflow(generations: int = 6) -> StdWorkflow:
    """PSO + Sphere with both history channels enabled (the readback targets)."""
    return StdWorkflow(
        make_pso(pop_size=_POP_SIZE, lb=_LB, ub=_UB),
        Sphere(),
        monitor=EvalMonitorConfig(full_fit_history=True, full_sol_history=True),
        opt_direction="min",
        num_generations=generations,
    )


def _run_histories(generations: int) -> tuple[StdWorkflow, EvalMonitorConfig]:
    """Run a fresh workflow for `generations` and return it with its config."""
    wf = _make_workflow(generations)
    wf.run(seed=42)
    return wf, wf.monitor_config


def _eager_readback(self: StdWorkflow, target: list, tensor) -> None:
    """Pre-pipeline ``_readback``: copy the monitored leaf immediately.

    Monkeypatched over the pipelined implementation to produce the reference
    histories the deferred path must match exactly.
    """
    if tensor is None:
        return
    target.append(tensor.to(_CPU).numpy())


class _FakeDevice:
    """Minimal stand-in for ``evoxc.core.Device`` carrying only ``kind``."""

    kind = "cuda"


class _FakeDeviceLeaf:
    """A device-resident monitor leaf; ``_readback`` only reads ``.device.kind``."""

    def __init__(self) -> None:
        self.device = _FakeDevice()


# --------------------------------------------------------- eager reference


def test_deferred_readback_matches_eager(monkeypatch):
    """The deferred pipeline must append exactly what an eager copy would."""
    gens = 6
    with monkeypatch.context() as patch:
        patch.setattr(StdWorkflow, "_readback", _eager_readback)
        eager_wf, eager_cfg = _run_histories(gens)
    assert eager_wf._pending_readbacks == []

    deferred_wf, deferred_cfg = _run_histories(gens)
    assert deferred_wf._pending_readbacks == []

    for cfg in (eager_cfg, deferred_cfg):
        assert len(cfg.fit_history) == gens
        assert len(cfg.sol_history) == gens
    for gen in range(gens):
        assert np.array_equal(eager_cfg.fit_history[gen], deferred_cfg.fit_history[gen])
        assert np.array_equal(eager_cfg.sol_history[gen], deferred_cfg.sol_history[gen])


# ------------------------------------------------------------ deferral itself


def test_readback_defers_device_leaf(monkeypatch):
    """A device leaf is parked in ``_pending_readbacks`` and appended on drain."""
    expected = np.asarray([1.5, -2.0, 3.25], dtype=np.float32)

    class _FakeHandle:
        def result(self):
            return evoxc.core.tensor(expected)

    sink_calls = []

    def _fake_sink(device_tensor):
        sink_calls.append(device_tensor)
        return _FakeHandle()

    monkeypatch.setattr(evoxc.io, "sink", _fake_sink)

    wf = _make_workflow()
    target: list = []
    leaf = _FakeDeviceLeaf()

    wf._readback(target, leaf)
    assert target == []  # nothing appended yet — the copy is only scheduled
    assert len(wf._pending_readbacks) == 1
    assert sink_calls == [leaf]

    wf._drain_readbacks()
    assert len(target) == 1
    assert np.array_equal(target[0], expected)
    assert wf._pending_readbacks == []


# -------------------------------------------------------------------- flush


def test_flush_idempotent():
    """`flush` on an already-drained workflow appends nothing further."""
    gens = 3
    wf = _make_workflow(gens)
    wf.run()
    cfg = wf.monitor_config
    assert len(cfg.fit_history) == gens

    wf.flush()
    wf.flush()
    assert len(cfg.fit_history) == gens
    assert wf._pending_readbacks == []
