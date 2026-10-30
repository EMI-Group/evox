# unit_test/ — test suite for the evox_etl release package

## Intent
Holds the test suite of the EvoX 2.0 pre-release, which is built on the `evoxc` compiler and ships the functional package `src/evox_etl/` (+ the `src/evox_etl_ext/` plugin namespace).
Only the canonical ETL suite `unit_test/etl/` remains; the old PyTorch 1.x test tree was removed with the torch package.
The suite is pure `pytest` and must not import torch (torch is no longer a dependency of this repository).

## API Surface
- `conftest.py` — repo-root bootstrap: inserts the repo root and `<repo>/src` into `sys.path` so the PEP 420 namespace packages `evox_etl` / `evox_etl_ext` import without installation.
- `etl/` — the whole suite, mirroring `src/evox_etl/` one directory per subpackage.
- `etl/test_no_torch.py` — the torch-free release gate (runs a fresh interpreter, imports `evox_etl`, runs a small PSO-on-Sphere optimization, then asserts no `torch` module entered `sys.modules`).

## Running
From the repository root (the `evoxc`/`numpy`/`pytest`/`plotly` venv):
```
PYTHONPATH=src <venv>/bin/python -m pytest -q
```
`[tool.pytest.ini_options] testpaths = ["unit_test/etl"]` in `pyproject.toml` makes that the full gate: 552 tests, ~2 min, 0 errors, 0 skips.
Per-suite runs (useful for faster failure isolation; all run from the repository root):
```
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/algorithms -q      # 182, ~80 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/operators -q       #  86, ~5 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/problems -q        # 144, ~30 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/metrics -q         #  50, ~5 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/vis_tools -q       #  42, ~1 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/workflows -q       #  41, ~5 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/ext -q             #   6, <1 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/test_no_torch.py -q #  1, <1 s
```
The counts sum to 552 with no basename collisions: every subdirectory that ships an `__init__.py` (`algorithms/mo/`, `algorithms/so/es_variants/`, `operators/`, `problems/`, `metrics/`, `vis_tools/`, `ext/`) imports its test modules with the package prefix, which keeps same-basename files distinct.

## Torch-free gate
- No file under `unit_test/` may import torch, `evox`, or any `src/evox` module — that tree no longer exists and torch is not installed as part of the 2.0 dependency set (`pyproject.toml`: `evoxc`, `numpy`).
- `etl/test_no_torch.py` enforces this in a fresh subprocess (`sys.executable -c ...`, `PYTHONPATH=src:<repo>`), so an in-process `sys.modules` check cannot be fooled by the pytest session.
- Plotly is an optional extra (`evox[vis]`): `vis_tools/` and `workflows/test_eval_monitor_plot.py` keep their figure assertions behind `skipif`, so a plotly-less environment still passes.

## Constraints
- Never modify `src/evox_etl/` or `src/evox_etl_ext/` from a test — if a test exposes a package bug, report it instead of fixing package code.
- New tests follow the existing pattern: `evoxc.build(fn, *specs, backend="numpy")` + `evoxc.run(exe, *args)`, numpy inputs/expectations, `.numpy()` on results (never `np.asarray`).
- Keep test files focused (<~400 lines) and avoid slow integration downloads; there is no CUDA or compiled-backend coverage in this tree (that lives in `benchmarks/etl_vs_torch/`).
- Basenames must stay unique in directories without an `__init__.py` (`etl/algorithms/`, `algorithms/so/de_variants/`, `algorithms/so/pso_variants/`, `etl/workflows/`), because those modules import as top-level names.

## Routing Table
| Path | Description |
|---|---|
| `./etl` | Canonical ETL suite for `src/evox_etl/` — algorithms (SO/MO), operators, problems (incl. CEC2022/DTLZ/HPO/virtual), metrics, vis_tools, workflows, extension autoload, and the torch-free gate |
