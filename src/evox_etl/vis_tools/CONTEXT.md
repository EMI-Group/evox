# evox_etl.vis_tools — host-side visualization utilities

## Intent
Host-side visualization for the functional ETL EvoX rewrite: interactive Plotly
figure builders (a near-verbatim functional port of the torch `evox.vis_tools`
package) plus the EvoXVision binary serialization format (`.exv`) streaming writer.
Nothing here is traced by ETL — it consumes already-materialized NumPy arrays (e.g.
host-side history lists collected by a workflow/monitor) and produces figures or
writes files. It is a drop-in replacement for `evox.vis_tools` (same symbol names,
parameter order, and Plotly figure structure).

## API Surface
- `plot.py` — six public figure builders, all returning `plotly.graph_objects.Figure`:
  `plot_dec_space(population_history, **kwargs)`,
  `plot_obj_space_1d(fitness_history, animation=True, **kwargs)`,
  `plot_obj_space_1d_no_animation(fitness_history, **kwargs)`,
  `plot_obj_space_1d_animation(fitness_history, **kwargs)`,
  `plot_obj_space_2d(fitness_history, problem_pf=None, sort_points=False, **kwargs)`,
  `plot_obj_space_3d(fitness_history, problem_pf=None, sort_points=False, **kwargs)`.
- `exv.py` — `new_exv_metadata(population1, population2, fitness1, fitness2) -> dict`,
  class `EvoXVisionAdapter(file_path, buffering=0)` with
  `set_metadata`, `write_header`, `write(*fields)`, `flush`
  (private `_get_data_type`, `_write_magic_number`, `_write_metedata`).
- `__init__.py` — re-exports the six plot functions + `new_exv_metadata` +
  `EvoXVisionAdapter`; `__all__` lists exactly those eight names. The private
  `_get_data_type` is intentionally NOT exported.

## Constraints
- **Optional Plotly**: `plot.py` guards the import at module level
  (`try: import plotly.graph_objects as go` / `except ImportError: go = None`).
  `import evox_etl.vis_tools` MUST succeed WITHOUT plotly installed; the `plot_*`
  functions call `_require_plotly()` first and raise an `ImportError` mentioning
  `pip install "evox[vis]"`. `from __future__ import annotations` is required so the
  `-> go.Figure` annotations don't evaluate when `go is None`. `exv.py` never needs
  plotly.
- **numpy/plotly/json allowed here** — documented exception to DESIGN.md §10's
  "no numpy imports in `src/evox_etl/**`": this subtree is host-side (like
  `problems/numerical/cec2022.py`), not graph code, so numpy/plotly/json/pathlib are
  expected and never traced by ETL.
- **Binary format (exv) preserved exactly**: magic `b"\x65\x78\x76\x31"` ("exv1"),
  then u32 little-endian metadata byte-length, then JSON-utf8 metadata, then raw
  binary chunks. Little-endian throughout.
- **`plot.py` is one 614-line faithful file** intentionally: it mirrors the
  589-line torch module name-for-name (plus ~25 lines of the plotly guard, the
  `_require_plotly` helper, and type hints) so it stays diffable against the
  reference. `plot.py` is the single source — do NOT add a second plot module
  unless you also re-export, since `from evox_etl.vis_tools.plot import
  plot_obj_space_2d` is part of the public surface.
- Style: `from __future__ import annotations`; type hints on all public functions;
  1-3 line docstrings; keep the upstream private name spellings (incl. the
  `_write_metedata` typo) for a faithful port.

## See Also
- Read-only torch reference (never modify): `../../evox/vis_tools/`
  (`plot.py`, `exv.py`).
- Tests (owned by another agent): `../unit_test/etl/vis_tools/`.
- Optional extra: `pyproject.toml` `[project.optional-dependencies] vis` (`plotly >= 5.0.0`).
