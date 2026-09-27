# unit_test/etl/problems — tests for the evox_etl problems

## Intent
Pytest suite for the functional `evox_etl` problems — the numerical problems
(`basic`, `dtlz`, `cec2022`), the neuroevolution virtual Gaussian-noise problem,
and the host-side HPO wrapper — running everything through `etl.build` +
`etl.run` on the numpy backend.

## Structure
- `test_basic.py` — pure-etl tests (NO torch) for the basic problems: shapes/
  dtypes, known optima, raw-math (no clamping) behaviour, shift/affine configs,
  `*_func` helpers, config validation.
- `test_dtlz.py` — pure-etl tests for DTLZ1-7: evaluate shapes, known values,
  custom configs, `pf` reference fronts (row counts + value ranges).
- `test_cec2022.py` — pure-etl tests for CEC2022: all valid (function, dim)
  pairs stay finite, config assertions, error paths, f1 shift point.
- `test_virtual_problem.py` — pure-etl tests for the neuroevolution virtual
  (training-free) Gaussian-noise problem: config normalization/validation, the
  perturbation paths (`lora_rank` full-noise vs low-rank), the evaluate/init-step
  contract, and convergence driven through `StdWorkflow`.
- `test_hpo_wrapper.py` — pure-etl tests for the host-side HPO wrapper:
  `evaluate` runs the inner `StdWorkflow` with the substituted hyperparameters
  (exact parity against a manually built workflow), the repeat/aggregation
  policy, multi-objective inner problems, monitor edge cases, and `random_search`.
- `parity/test_parity.py` — the ONLY file allowed to import torch; compares
  `evox_etl` results against the torch `evox` reference on seeded populations.

## Packaging
- `__init__.py` in both `problems/` and `problems/parity/` makes them distinct
  pytest packages. Without these, `test_parity.py` here would collide (module
  name `parity.test_parity`) with `unit_test/etl/{algorithms,operators,metrics}`
  parity suites when the suites run together.
- `conftest.py` shims `sys.path` (repo root + `src/`) for the uninstalled PEP
  420 `evox_etl` namespace package. It locates the repo root by searching
  parents of `__file__` for `pyproject.toml` and is idempotent (no duplicate
  sys.path entries).

## Run
```
/mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest unit_test/etl/problems -q
```
192 tests, ~100s (parity imports torch). Runs from the repository root.

## Constraints
- Never modify `src/evox_etl/` from here — if a test exposes a package bug,
  report it instead of fixing package code.
- Pure-etl files must not import torch; torch is only allowed in `parity/`.
