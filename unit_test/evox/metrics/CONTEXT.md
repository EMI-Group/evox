# unit_test/evox/metrics — tests for the evox metrics

## Intent
Pytest suite for the functional `evox` metrics (`gd`, `igd`, `gd_plus`,
`igd_plus`, `hv` / `bounding_cube_monte_carlo_hv`, `each_cube_monte_carlo_hv`),
running everything through `evoxc.build` + `evoxc.run` on the numpy backend.

## Structure
- `test_metrics.py` — metrics vs hand-rolled numpy references (GD, IGD, GD+,
  IGD+ for several shapes/seeds and p exponents) and Monte-Carlo HV vs
  brute-force recomputation from the drawn samples.
  Also the HV reference-point CLIPPING regressions for both estimators
  (`hv` / `bounding_cube_monte_carlo_hv` and `each_cube_monte_carlo_hv`):
  a point that is not better than `ref` along an objective has zero extent
  there, so it adds no hypervolume — pinned by an exact `== 0.0` case, the
  0.32 two-point half-outside front (`num_sample=400000`), and a 3-objective
  mixed front where the exact union volume comes from host-side
  inclusion-exclusion and adding the outside point must not change the value.

## Packaging
- `__init__.py` in `metrics/` keeps its test module distinct from same-named
  modules in other suites when everything runs together.
- `conftest.py` shims `sys.path` (repo root + `src/`) so the `evox`
  package under `src/` imports without an installation. It locates the repo
  root by searching parents of `__file__` for `pyproject.toml` and is
  idempotent (no duplicate sys.path entries).

## Run
```
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/evox/metrics -q
```
50 tests, ~5s. Runs from the repository root.

## Constraints
- Never modify `src/evox/` from here — if a test exposes a package bug,
  report it instead of fixing package code.
- No test file in this tree may pull in torch; references are plain numpy.
