# unit_test/etl/metrics — tests for the evox_etl metrics

## Intent
Pytest suite for the functional `evox_etl` metrics (`gd`, `igd`, `gd_plus`,
`igd_plus`, `hv` / `bounding_cube_monte_carlo_hv`, `each_cube_monte_carlo_hv`),
running everything through `etl.build` + `etl.run` on the numpy backend.
Relocated from `src/evox_etl/metrics/tests/` (the old location is kept in sync
by another agent until deletion).

## Structure
- `test_metrics.py` — pure-etl tests (NO torch): metrics vs hand-rolled numpy
  references (GD, IGD, GD+, IGD+ for several shapes/seeds and p exponents) and
  Monte-Carlo HV vs brute-force recomputation from the drawn samples.
- `parity/test_parity.py` — the ONLY file allowed to import torch; compares
  `evox_etl` `gd`/`igd` against the torch `evox` reference on seeded inputs.

## Packaging
- `__init__.py` in both `metrics/` and `metrics/parity/` makes them distinct
  pytest packages. Without these, `test_parity.py` here would collide (module
  name `parity.test_parity`) with `unit_test/etl/{algorithms,operators,problems}`
  parity suites when the suites run together.
- `conftest.py` shims `sys.path` (repo root + `src/`) for the uninstalled PEP
  420 `evox_etl` namespace package. It locates the repo root by searching
  parents of `__file__` for `pyproject.toml` and is idempotent (no duplicate
  sys.path entries).

## Run
```
/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl/metrics -q
```
53 tests, ~10s (parity imports torch). Runs from the repository root.

## Constraints
- Never modify `src/evox_etl/` from here — if a test exposes a package bug,
  report it instead of fixing package code.
- Pure-etl files must not import torch; torch is only allowed in `parity/`.
