# unit_test/ — test suite for the evox release package

## Intent
Holds the test suite of the EvoX 2.0 pre-release, which is built on the `evoxc` compiler and ships the functional package `src/evox/` (+ the `src/evox_ext/` plugin namespace).
Only the canonical EvoX suite `unit_test/evox/` remains; the old PyTorch 1.x test tree was removed with the torch package.
The suite is pure `pytest` and must stay torch-free (torch is no longer a dependency of this repository).

## API Surface
- `conftest.py` — repo-root bootstrap: inserts the repo root and `<repo>/src` into `sys.path` so the PEP 420 namespace packages `evox` / `evox_ext` import without installation.
- `evox/` — the whole suite, mirroring `src/evox/` one directory per subpackage.
- `evox/test_no_torch.py` — the torch-free release gate (runs a fresh interpreter, imports `evox`, runs a small PSO-on-Sphere optimization, then asserts no `torch` module entered `sys.modules`).

## Running
From the repository root (the `evoxc`/`numpy`/`pytest`/`plotly` venv):
```
PYTHONPATH=src <venv>/bin/python -m pytest -q
```
`[tool.pytest.ini_options] testpaths = ["unit_test/evox"]` in `pyproject.toml` makes that the full gate: 552 tests, ~2 min, 0 errors, 0 skips.
Per-suite runs (useful for faster failure isolation; all run from the repository root):
```
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/evox/algorithms -q      # 182, ~80 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/evox/operators -q       #  86, ~5 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/evox/problems -q        # 144, ~30 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/evox/metrics -q         #  50, ~5 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/evox/vis_tools -q       #  42, ~1 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/evox/workflows -q       #  41, ~5 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/evox/ext -q             #   6, <1 s
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/evox/test_no_torch.py -q #  1, <1 s
```
The counts sum to 552 with no basename collisions: every subdirectory that ships an `__init__.py` (`algorithms/mo/`, `algorithms/so/es_variants/`, `operators/`, `problems/`, `metrics/`, `vis_tools/`, `ext/`) imports its test modules with the package prefix, which keeps same-basename files distinct.

## Torch-free gate
- No file under `unit_test/` may pull in torch or the removed 1.x torch API — the suite imports the functional `evox` package only, and torch is not part of the 2.0 dependency set (`pyproject.toml`: `evoxc`, `numpy`).
- `evox/test_no_torch.py` enforces this in a fresh subprocess (`sys.executable -c ...`, `PYTHONPATH=src:<repo>`), so an in-process `sys.modules` check cannot be fooled by the pytest session.
- Plotly is an optional extra (`evox[vis]`): `vis_tools/` and `workflows/test_eval_monitor_plot.py` keep their figure assertions behind `skipif`, so a plotly-less environment still passes.

## Constraints
- Never modify `src/evox/` or `src/evox_ext/` from a test — if a test exposes a package bug, report it instead of fixing package code.
- New tests follow the existing pattern: `evoxc.build(fn, *specs, backend="numpy")` + `evoxc.run(exe, *args)`, numpy inputs/expectations, `.numpy()` on results (never `np.asarray`).
- Keep test files focused (<~400 lines) and avoid slow integration downloads; there is no CUDA or compiled-backend coverage in this tree (that lives in `benchmarks/evox_vs_torch/`).
- Basenames must stay unique in directories without an `__init__.py` (`evox/algorithms/`, `algorithms/so/de_variants/`, `algorithms/so/pso_variants/`, `evox/workflows/`), because those modules import as top-level names.

## Routing Table
| Path | Description |
|---|---|
| `./evox` | Canonical EvoX suite for `src/evox/` — algorithms (SO/MO), operators, problems (incl. CEC2022/DTLZ/HPO/virtual), metrics, vis_tools, workflows, extension autoload, and the torch-free gate |
