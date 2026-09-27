"""End-to-end tests for the auxiliary-history channel of StdWorkflow/EvalMonitor.

Auxiliary-history channel (torch parity): an algorithm module MAY define an
optional PLAIN host-side function ``record_step(config, state, candidate,
fitness) -> dict[str, etl_tensor]``. ``StdWorkflow`` calls it after each
generation (host-side, never inside a trace) when both the algorithm module
defines ``record_step`` AND the monitor module defines ``record_auxiliary``,
then hands the returned dict to the monitor. ``EvalMonitor`` stores it in
``EvalMonitorConfig.aux_history`` (gated on ``full_pop_history``) and prefers
``aux_history["pop"]`` for its ``pop_history`` property.

These tests drive that whole path through ``evox_etl.workflows.StdWorkflow``
with toy components defined in SEPARATE sibling modules (the workflow resolves
each component's plain functions via ``type(config).__module__``, so two
configs in one module would collide on names like ``init``):

* ``aux_toy_algorithm.py``          — toy algorithm WITH ``record_step``
* ``aux_toy_algorithm_no_hook.py``  — toy algorithm WITHOUT ``record_step``
                                      (the back-compat / legacy path)
* ``aux_toy_problem.py``            — stateless toy sphere problem

Numpy backend only; no torch imports.
"""

from __future__ import annotations

import aux_toy_algorithm
import aux_toy_algorithm_no_hook
import aux_toy_problem
import etl
import etl.core
import numpy as np

import evox_etl.workflows.eval_monitor as em
from evox_etl.workflows import EvalMonitorConfig, StdWorkflow

POP, DIM, GENS = 8, 3, 6

# ----------------------------------------------------------------- helpers


def _workflow(algo_cfg, *, full_pop_history: bool, num_generations: int = GENS, **monitor_kwargs):
    """Build a StdWorkflow over the toy algorithm/problem pair.

    ``algo_cfg`` selects the hook (``aux_toy_algorithm``) or no-hook
    (``aux_toy_algorithm_no_hook``) toy algorithm — the config's module decides
    which plain functions the workflow resolves.
    """
    return StdWorkflow(
        algo_cfg,
        aux_toy_problem.AuxToySphere(dim=DIM),
        monitor=EvalMonitorConfig(full_pop_history=full_pop_history, **monitor_kwargs),
        opt_direction="min",
        num_generations=num_generations,
    )


def _hook_cfg(pop_size: int = POP, dim: int = DIM):
    return aux_toy_algorithm.AuxToyAlgorithmConfig(pop_size=pop_size, dim=dim)


def _no_hook_cfg(pop_size: int = POP, dim: int = DIM):
    return aux_toy_algorithm_no_hook.AuxToyNoHookConfig(pop_size=pop_size, dim=dim)


# ------------------------------------------------- aux channel: keys, shapes


def test_aux_history_records_each_key_once_per_generation():
    """``record_step``'s keys land in ``aux_history`` with one entry/generation."""
    wf = _workflow(_hook_cfg(), full_pop_history=True)
    wf.run()

    aux = wf.monitor.aux_history
    assert set(aux) == {"center", "pop", "fit"}
    for key in ("center", "pop", "fit"):
        assert len(aux[key]) == GENS
        assert all(isinstance(entry, np.ndarray) for entry in aux[key])

    # Shapes: center is the (dim,) population mean, "pop" the raw (pop_size, dim)
    # candidate batch and "fit" its rank-1 (pop_size,) fitness.
    assert all(entry.shape == (DIM,) for entry in aux["center"])
    assert all(entry.shape == (POP, DIM) for entry in aux["pop"])
    assert all(entry.shape == (POP,) for entry in aux["fit"])


def test_auxiliary_history_alias_and_config_identity():
    """``auxiliary_history`` aliases ``aux_history`` (same config dict object)."""
    wf = _workflow(_hook_cfg(), full_pop_history=True)
    wf.run()

    mon = wf.monitor
    assert mon.auxiliary_history is mon.aux_history
    assert mon.aux_history is wf.monitor_config.aux_history


def test_aux_entries_track_monitor_stored_batches():
    """Aux "fit"/"pop" mirror the monitor's own latest batch / history."""
    wf = _workflow(_hook_cfg(), full_pop_history=True, full_sol_history=True)
    wf.run()

    mon = wf.monitor
    aux = mon.aux_history

    # "fit" is exactly the monitor's per-generation (un-negated) fitness history
    fit_history = mon.get_fitness_history()
    assert len(fit_history) == GENS
    for recorded, hist in zip(aux["fit"], fit_history):
        assert recorded.shape == hist.shape == (POP,)
        assert np.allclose(recorded, hist)

    # the LAST "fit"/"pop" entries are the monitor's stored latest batch
    latest_solution = mon.get_latest_solution()
    assert latest_solution.shape == (POP, DIM)
    assert np.allclose(aux["pop"][-1], latest_solution)
    assert np.allclose(aux["fit"][-1], mon.get_latest_fitness())

    # "pop" is the raw candidate batch — identical to the solution history
    sol_history = mon.solution_history
    assert len(sol_history) == GENS
    for recorded, sol in zip(aux["pop"], sol_history):
        assert recorded.shape == sol.shape == (POP, DIM)
        assert np.allclose(recorded, sol)

    # "center" is the (dim,) mean the algorithm carried in its state
    assert all(np.all(np.isfinite(entry)) for entry in aux["center"])


def test_aux_channel_also_runs_under_fit():
    """``wf.fit()`` (no generation-0 fitness arg) records the same aux history."""
    wf = _workflow(_hook_cfg(), full_pop_history=True)
    best = wf.fit()

    assert np.isfinite(best) and best >= 0.0  # sphere is non-negative
    aux = wf.monitor.aux_history
    assert set(aux) == {"center", "pop", "fit"}
    assert all(len(aux[key]) == GENS for key in ("center", "pop", "fit"))


# -------------------------------------------------- pop_history from aux chan


def test_pop_history_prefers_aux_pop_channel():
    """``pop_history`` IS ``aux_history["pop"]`` when a "pop" key was recorded."""
    wf = _workflow(_hook_cfg(), full_pop_history=True)
    wf.run()

    mon = wf.monitor
    aux_pop = wf.monitor_config.aux_history["pop"]
    assert mon.pop_history is aux_pop
    assert len(mon.pop_history) == GENS
    assert all(entry.shape == (POP, DIM) for entry in mon.pop_history)
    # no double-record: the legacy list stays untouched when the hook is present
    assert wf.monitor_config.pop_history == []


# ------------------------------------------------------ full_pop_history=False


def test_aux_history_empty_when_full_pop_history_off():
    """``full_pop_history=False`` gates the whole aux channel off."""
    wf = _workflow(_hook_cfg(), full_pop_history=False)
    wf.run()

    mon = wf.monitor
    assert mon.aux_history == {}
    assert mon.auxiliary_history == {}
    assert wf.monitor_config.aux_history == {}
    # no aux "pop" key -> pop_history falls back to the (empty) legacy list
    assert mon.pop_history == []
    assert wf.monitor_config.pop_history == []
    # the pre-existing fit history is unaffected by the aux gate
    assert len(mon.get_fitness_history()) == GENS


# ------------------------------------------------- back-compat: no record_step


def test_no_hook_algorithm_keeps_legacy_pop_history():
    """Without ``record_step`` the workflow appends latest_solution (legacy)."""
    assert not hasattr(aux_toy_algorithm_no_hook, "record_step")
    wf = _workflow(_no_hook_cfg(), full_pop_history=True)
    wf.run()

    mon = wf.monitor
    assert mon.aux_history == {}
    legacy = wf.monitor_config.pop_history
    assert len(legacy) == GENS
    assert all(entry.shape == (POP, DIM) for entry in legacy)
    # no aux "pop" key -> pop_history falls back to the legacy list
    assert mon.pop_history is legacy
    assert np.allclose(legacy[-1], mon.get_latest_solution())


# ------------------------------------- module-level record_auxiliary edge cases


def test_module_record_auxiliary_noop_when_gate_off():
    """The module-level hook early-returns when ``full_pop_history`` is off."""
    cfg = EvalMonitorConfig(full_pop_history=False)
    tensor = etl.core.tensor(np.arange(DIM, dtype=np.float32))
    em.record_auxiliary(cfg, {"x": tensor})
    assert cfg.aux_history == {}


def test_module_record_auxiliary_converts_and_accumulates():
    """Each call appends one CPU-numpy copy per key (channel accumulates)."""
    cfg = EvalMonitorConfig(full_pop_history=True)
    first = etl.core.tensor(np.zeros(DIM, dtype=np.float32))
    second = etl.core.tensor(np.ones(DIM, dtype=np.float32))
    em.record_auxiliary(cfg, {"x": first})
    em.record_auxiliary(cfg, {"x": second, "y": first})
    assert set(cfg.aux_history) == {"x", "y"}
    assert len(cfg.aux_history["x"]) == 2
    assert len(cfg.aux_history["y"]) == 1
    assert np.allclose(cfg.aux_history["x"][0], np.zeros(DIM, dtype=np.float32))
    assert np.allclose(cfg.aux_history["x"][1], np.ones(DIM, dtype=np.float32))
