"""Tests for :mod:`evox_etl.vis_tools.plot` — the six Plotly figure builders.

Every public builder must return a ``plotly.graph_objects.Figure`` from small,
deterministic random histories, cover the ``problem_pf`` / ``sort_points`` /
``animation`` variants, forward ``**kwargs`` to the layout where the reference
implementation does, and raise a clear ``ImportError`` when the optional plotly
dependency is unavailable.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pytest

go = pytest.importorskip("plotly.graph_objects")

import evox_etl.vis_tools.plot as _plot  # noqa: E402
from evox_etl.vis_tools import (  # noqa: E402
    plot_dec_space,
    plot_obj_space_1d,
    plot_obj_space_1d_animation,
    plot_obj_space_1d_no_animation,
    plot_obj_space_2d,
    plot_obj_space_3d,
)

_SEED = 1234
_N_GENS = 3
_POP = 6

_rng = np.random.default_rng(_SEED)
DEC_HISTORY: List[np.ndarray] = [_rng.random((_POP, 2)) for _ in range(_N_GENS)]
FIT_1D: List[np.ndarray] = [_rng.random(_POP) for _ in range(_N_GENS)]
FIT_2D: List[np.ndarray] = [_rng.random((_POP, 2)) for _ in range(_N_GENS)]
FIT_3D: List[np.ndarray] = [_rng.random((_POP, 3)) for _ in range(_N_GENS)]
PF_2D: np.ndarray = _rng.random((10, 2))
PF_3D: np.ndarray = _rng.random((10, 3))


class TestFigureTypes:
    """Each builder returns a plotly ``Figure``."""

    def test_plot_dec_space(self) -> None:
        assert isinstance(plot_dec_space(DEC_HISTORY), go.Figure)

    def test_plot_obj_space_1d_animation_true(self) -> None:
        assert isinstance(plot_obj_space_1d(FIT_1D, animation=True), go.Figure)

    def test_plot_obj_space_1d_animation_false(self) -> None:
        assert isinstance(plot_obj_space_1d(FIT_1D, animation=False), go.Figure)

    def test_plot_obj_space_1d_animation_fn(self) -> None:
        assert isinstance(plot_obj_space_1d_animation(FIT_1D), go.Figure)

    def test_plot_obj_space_1d_no_animation_fn(self) -> None:
        assert isinstance(plot_obj_space_1d_no_animation(FIT_1D), go.Figure)

    def test_plot_obj_space_2d_without_pf(self) -> None:
        assert isinstance(plot_obj_space_2d(FIT_2D), go.Figure)

    def test_plot_obj_space_2d_with_pf(self) -> None:
        assert isinstance(plot_obj_space_2d(FIT_2D, problem_pf=PF_2D), go.Figure)

    def test_plot_obj_space_2d_sort_points(self) -> None:
        assert isinstance(plot_obj_space_2d(FIT_2D, problem_pf=PF_2D, sort_points=True), go.Figure)

    def test_plot_obj_space_3d_without_pf(self) -> None:
        assert isinstance(plot_obj_space_3d(FIT_3D), go.Figure)

    def test_plot_obj_space_3d_with_pf(self) -> None:
        assert isinstance(plot_obj_space_3d(FIT_3D, problem_pf=PF_3D), go.Figure)

    def test_plot_obj_space_3d_sort_points(self) -> None:
        assert isinstance(plot_obj_space_3d(FIT_3D, problem_pf=PF_3D, sort_points=True), go.Figure)


class TestParetoFrontTrace:
    """``problem_pf`` adds a "Pareto Front" trace, absence omits it."""

    def test_2d_pf_trace_present(self) -> None:
        fig = plot_obj_space_2d(FIT_2D, problem_pf=PF_2D)
        names = [trace.name for trace in fig.data]
        assert "Pareto Front" in names

    def test_2d_no_pf_trace_absent(self) -> None:
        fig = plot_obj_space_2d(FIT_2D, problem_pf=None)
        names = [trace.name for trace in fig.data]
        assert "Pareto Front" not in names

    def test_3d_pf_trace_present(self) -> None:
        fig = plot_obj_space_3d(FIT_3D, problem_pf=PF_3D)
        names = [trace.name for trace in fig.data]
        assert "Pareto Front" in names

    def test_3d_no_pf_trace_absent(self) -> None:
        fig = plot_obj_space_3d(FIT_3D, problem_pf=None)
        names = [trace.name for trace in fig.data]
        assert "Pareto Front" not in names


class TestKwargsPassThrough:
    """``**kwargs`` reach the plotly layout for builders that forward them.

    ``plot_obj_space_1d_no_animation`` deliberately does NOT forward kwargs —
    this mirrors the torch reference (``src/evox/vis_tools/plot.py``), so it is
    intentionally excluded here.
    """

    def test_dec_space(self) -> None:
        fig = plot_dec_space(DEC_HISTORY, title="dec-title")
        assert fig.layout.title.text == "dec-title"

    def test_obj_space_1d_animation(self) -> None:
        fig = plot_obj_space_1d(FIT_1D, animation=True, title="1d-title")
        assert fig.layout.title.text == "1d-title"

    def test_obj_space_1d_animation_fn(self) -> None:
        fig = plot_obj_space_1d_animation(FIT_1D, title="1d-anim-title")
        assert fig.layout.title.text == "1d-anim-title"

    def test_obj_space_2d(self) -> None:
        fig = plot_obj_space_2d(FIT_2D, problem_pf=PF_2D, title="2d-title")
        assert fig.layout.title.text == "2d-title"

    def test_obj_space_3d(self) -> None:
        fig = plot_obj_space_3d(FIT_3D, problem_pf=PF_3D, title="3d-title")
        assert fig.layout.title.text == "3d-title"


_PUBLIC_PLOT_CALLS: List[Callable[[], object]] = [
    lambda: plot_dec_space(DEC_HISTORY),
    lambda: plot_obj_space_1d(FIT_1D),
    lambda: plot_obj_space_1d_animation(FIT_1D),
    lambda: plot_obj_space_1d_no_animation(FIT_1D),
    lambda: plot_obj_space_2d(FIT_2D),
    lambda: plot_obj_space_3d(FIT_3D),
]


class TestPlotlyMissing:
    """Without plotly, every public builder raises a helpful ImportError."""

    @pytest.mark.parametrize("call", _PUBLIC_PLOT_CALLS)
    def test_raises_import_error(self, monkeypatch: pytest.MonkeyPatch, call: Callable[[], object]) -> None:
        monkeypatch.setattr(_plot, "go", None)
        with pytest.raises(ImportError) as excinfo:
            call()
        assert "evox[vis]" in str(excinfo.value)
