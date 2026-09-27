# unit_test/etl/vis_tools — vis_tools suite
## Intent
pytest suite for `evox_etl.vis_tools` (Plotly figure builders + the EvoXVision
`.exv` binary serialization format). Numpy/plotly only — no torch. Runnable
without installing `evox_etl` (PEP 420 namespace package under `src/`).
## Files
- `conftest.py` — sys.path shim (repo root + `src/`), idempotent, locations
  computed from `__file__`.
- `__init__.py` — empty package marker (mirrors `metrics/`, `operators/`,
  `problems/`; there is no sibling helper that must stay a top-level module).
- `test_exv.py` — `.exv` dtype/metadata/round-trip (`_get_data_type`,
  `new_exv_metadata`, `EvoXVisionAdapter` write/flush).
- `test_plot.py` — the six figure builders plus the plotly-missing path.
## Constraints
- Figure assertions are `skipif`-guarded on plotly; the plotly-missing branch
  works without it.
- Run: `PYTHONPATH=src:<site-packages> <venv>/bin/python -m pytest
  unit_test/etl/vis_tools -q` (42 tests).
