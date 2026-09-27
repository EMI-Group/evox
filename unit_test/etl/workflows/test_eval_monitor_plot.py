"""Tests for the implemented ``EvalMonitor.plot`` (host-side Plotly figures).

``EvalMonitor.plot(problem_pf=None, source="eval", **kwargs)`` is a plain
host-side method (torch-parity port of ``evox.workflows.EvalMonitor.plot``) that
dispatches to the ``evox_etl.vis_tools`` figure builders by number of
objectives: 1 objective -> ``plot_obj_space_1d``, 2 -> ``plot_obj_space_2d``,
3 -> ``plot_obj_space_3d``, anything else -> warn + None. It returns None (with
a warning) when no history was recorded or when Plotly is unavailable.

The tests drive REAL ``StdWorkflow`` runs (numpy backend, no torch imports) and
call ``plot`` on the resulting monitor. The POP source uses the toy algorithm's
``record_step`` aux channel, so the toy modules are imported as TOP-LEVEL modules
(this directory intentionally has no ``__init__.py``, same as the sibling
``test_aux_history.py``).
"""

from __future__ import annotations

import aux_toy_algorithm
import aux_toy_problem
import numpy as np
import pytest

try:  # Plotly is an OPTIONAL dependency of evox_etl (extra "vis").
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised only without the optional extra
    go = None

import evox_etl.vis_tools.plot as vis_plot
from evox_etl.algorithms import make_nsga2, make_pso
from evox_etl.problems.numerical.basic import Sphere
from evox_etl.problems.numerical.dtlz import DTLZ2
from evox_etl.workflows import EvalMonitor, EvalMonitorConfig, StdWorkflow

# Figure-asserting tests need the optional Plotly extra; the warn/None paths do not.
requires_plotly = pytest.mark.skipif(go is None, reason='plotly is not installed (extra "evox[vis]")')

DIM = 4
LB, UB = [-5.0] * DIM, [5.0] * DIM
SO_GENS = 4


# ------------------------------------------------------------------ helpers


def _so_workflow(*, opt_direction: str = "min", gens: int = SO_GENS, **monitor_kwargs) -> StdWorkflow:
    """PSO + Sphere (1-D fitness) over a handful of generations."""
    return StdWorkflow(
        make_pso(pop_size=20, lb=LB, ub=UB),
        Sphere(),
        monitor=EvalMonitorConfig(**monitor_kwargs),
        opt_direction=opt_direction,
        num_generations=gens,
    )


def _mo_workflow(n_objs: int, *, d: int = 6, ref_num: int = 16, pop: int = 12, gens: int = 3) -> StdWorkflow:
    """NSGA2 + DTLZ2 (``n_objs``-dimensional fitness); small ``ref_num`` keeps it fast."""
    return StdWorkflow(
        make_nsga2(pop_size=pop, n_objs=n_objs, lb=[0.0] * d, ub=[1.0] * d),
        DTLZ2(d=d, m=n_objs, ref_num=ref_num),
        monitor=EvalMonitorConfig(full_fit_history=True),
        opt_direction=["min"] * n_objs,
        num_generations=gens,
    )


def _toy_workflow(*, opt_direction: str = "min", gens: int = 4, **monitor_kwargs) -> StdWorkflow:
    """Toy random-search algorithm (WITH the ``record_step`` hook) + toy sphere."""
    return StdWorkflow(
        aux_toy_algorithm.AuxToyAlgorithmConfig(pop_size=8, dim=3),
        aux_toy_problem.AuxToySphere(dim=3),
        monitor=EvalMonitorConfig(**monitor_kwargs),
        opt_direction=opt_direction,
        num_generations=gens,
    )


# --------------------------------------------------------------- SO (1 object)


@requires_plotly
def test_single_objective_eval_source_returns_1d_figure():
    wf = _so_workflow(full_fit_history=True)
    wf.run()

    fig = wf.monitor.plot()
    assert isinstance(fig, go.Figure)
    # plot_obj_space_1d draws the per-generation min/max/median/average traces
    assert [trace.name for trace in fig.data] == ["Min", "Max", "Median", "Average"]
    assert len(fig.data[0].y) == SO_GENS
    assert len(fig.frames) == SO_GENS  # animation=True is the 1-D builder's default


@requires_plotly
def test_single_objective_forwards_kwargs_to_plot_builder():
    wf = _so_workflow(full_fit_history=True)
    wf.run()

    fig = wf.monitor.plot(animation=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].y) == SO_GENS
    assert not fig.frames  # animation=False selects the no-animation builder


@requires_plotly
def test_eval_source_un_negates_maximization_history():
    """The "eval" source plots what the problem sees (opt-direction un-negated)."""
    wf = _so_workflow(opt_direction="max", full_fit_history=True)
    wf.run()

    # stored fitness is negated internally for maximization ...
    assert all(np.all(np.asarray(f) <= 0.0) for f in wf.monitor.fitness_history)
    # ... and plot(source="eval") un-negates it via get_fitness_history()
    fig = wf.monitor.plot(source="eval")
    assert isinstance(fig, go.Figure)
    assert np.max(fig.data[1].y) > 0.0  # data[1] is the "Max" trace


# ------------------------------------------------------------ MO (2/3 objects)


@requires_plotly
def test_three_objective_eval_source_returns_3d_figure():
    wf = _mo_workflow(3, d=12, ref_num=64)
    wf.run()

    assert all(np.asarray(f).shape == (12, 3) for f in wf.monitor.get_fitness_history())
    fig = wf.monitor.plot()
    assert isinstance(fig, go.Figure)
    assert all(len(trace.z) == 12 for trace in fig.data)  # 3-D scatter uses z


@requires_plotly
def test_three_objective_problem_pf_accepted():
    wf = _mo_workflow(3, d=12, ref_num=64)
    wf.run()

    problem_pf = np.random.default_rng(0).random((10, 3)).astype(np.float32)
    fig = wf.monitor.plot(problem_pf)
    assert isinstance(fig, go.Figure)
    # the overlay shows up in every frame alongside the population
    assert len(fig.frames[0].data) == 2


@requires_plotly
def test_two_objective_eval_source_returns_2d_figure():
    wf = _mo_workflow(2, d=6, ref_num=16)
    wf.run()

    assert all(np.asarray(f).shape == (12, 2) for f in wf.monitor.get_fitness_history())
    fig = wf.monitor.plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == 12  # 2-D scatter uses x/y


@requires_plotly
def test_two_objective_problem_pf_accepted():
    wf = _mo_workflow(2, d=6, ref_num=16)
    wf.run()

    problem_pf = np.random.default_rng(1).random((10, 2)).astype(np.float32)
    fig = wf.monitor.plot(problem_pf)
    assert isinstance(fig, go.Figure)
    assert len(fig.frames[0].data) == 2


# --------------------------------------------------------------- POP source


@requires_plotly
def test_pop_source_plots_aux_fit_channel():
    wf = _toy_workflow(full_pop_history=True)
    wf.run()

    assert set(wf.monitor.aux_history) == {"center", "pop", "fit"}
    assert all(np.asarray(f).shape == (8,) for f in wf.monitor.aux_history["fit"])

    fig = wf.monitor.plot(source="pop")
    assert isinstance(fig, go.Figure)
    assert [trace.name for trace in fig.data] == ["Min", "Max", "Median", "Average"]


@requires_plotly
def test_pop_source_uses_raw_aux_values_not_un_negated():
    """The "pop" source plots what the algorithm saw: raw values, NOT un-negated."""
    wf = _toy_workflow(opt_direction="max", full_pop_history=True)
    wf.run()

    raw = np.concatenate(wf.monitor.aux_history["fit"])
    assert np.all(raw <= 0.0)  # maximization is negated internally

    pop_fig = wf.monitor.plot(source="pop")
    eval_fig = wf.monitor.plot(source="eval")
    assert isinstance(pop_fig, go.Figure) and isinstance(eval_fig, go.Figure)
    assert np.max(pop_fig.data[1].y) <= 0.0 < np.max(eval_fig.data[1].y)


# ------------------------------------------------------------- error / warn


@requires_plotly
def test_invalid_source_raises_value_error():
    wf = _so_workflow(full_fit_history=True)
    wf.run()

    with pytest.raises(ValueError, match="Invalid source argument: nope, expect 'eval' or 'pop'."):
        wf.monitor.plot(source="nope")


def test_no_history_warns_and_returns_none():
    """``not fitness_history and not aux_history`` short-circuits to None."""
    mon = EvalMonitor(config=EvalMonitorConfig(), state=None)
    with pytest.warns(UserWarning, match="No fitness history recorded, return None"):
        assert mon.plot() is None


def test_workflow_without_recording_any_history_warns_and_returns_none():
    wf = _so_workflow(full_fit_history=False)
    wf.run()

    mon = wf.monitor
    assert mon.fitness_history == [] and mon.aux_history == {}
    with pytest.warns(UserWarning, match="No fitness history recorded, return None"):
        assert mon.plot() is None


def test_missing_plotly_warns_and_returns_none(monkeypatch):
    """Plotly absent (``vis_tools.plot.go is None``) => warn + None, not a raise."""
    monkeypatch.setattr(vis_plot, "go", None)
    wf = _so_workflow(full_fit_history=True)
    wf.run()
    assert wf.monitor.fitness_history  # history exists, so the vis check is reached

    with pytest.warns(UserWarning, match='No visualization tool available, return None. Hint: pip install "evox\\[vis\\]"'):
        assert wf.monitor.plot() is None


@requires_plotly
def test_four_objective_history_warns_not_supported():
    cfg = EvalMonitorConfig(fit_history=[np.zeros((4, 4), dtype=np.float32)])
    mon = EvalMonitor(config=cfg, state=None)

    with pytest.warns(UserWarning, match="Not supported yet."):
        assert mon.plot() is None
