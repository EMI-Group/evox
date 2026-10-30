# unit_test/etl/problems — tests for the evox_etl problems

## Intent
Pytest suite for the functional `evox_etl` problems — the numerical problems
(`basic`, `dtlz`, `cec2022`), the neuroevolution virtual Gaussian-noise problem,
and the host-side HPO wrapper — running everything through `evoxc.build` +
`evoxc.run` on the numpy backend.

## Structure
- `test_basic.py` — pure-ETL tests for the basic problems: shapes/
  dtypes, known optima, raw-math (no clamping) behaviour, shift/affine configs,
  `*_func` helpers, config validation.
- `test_dtlz.py` — pure-ETL tests for DTLZ1-7: evaluate shapes, known values,
  custom configs, `pf` reference fronts (row counts + value ranges).
- `test_cec2022.py` — pure-ETL tests for CEC2022: all valid (function, dim)
  pairs stay finite, config assertions, error paths, f1 shift point. It reads
  the shipped data through the module's own `DATA_DIR`
  (`src/evox_etl/problems/numerical/cec2022_input_data/`), so it also works
  from an installed wheel.
- `test_virtual_problem.py` — pure-ETL tests for the neuroevolution virtual
  (training-free) Gaussian-noise problem: config normalization/validation, the
  perturbation paths (`lora_rank` full-noise vs low-rank), the evaluate/init-step
  contract, and convergence driven through `StdWorkflow`.
- `test_hpo_wrapper.py` — pure-ETL tests for the host-side HPO wrapper:
  `evaluate` runs the inner `StdWorkflow` with the substituted hyperparameters
  (exact parity against a manually built workflow), the repeat/aggregation
  policy, multi-objective inner problems, monitor edge cases, and `random_search`.

## Packaging
- `__init__.py` in `problems/` makes it a distinct pytest package, so its test
  modules cannot collide with same-basename files in sibling suites.
- `conftest.py` shims `sys.path` (repo root + `src/`) for the uninstalled PEP
  420 `evox_etl` namespace package. It locates the repo root by searching
  parents of `__file__` for `pyproject.toml` and is idempotent (no duplicate
  sys.path entries).

## Run
```
PYTHONPATH=src <venv>/bin/python -m pytest unit_test/etl/problems -q
```
144 tests, ~30s. Runs from the repository root.

## Constraints
- Never modify `src/evox_etl/` from here — if a test exposes a package bug,
  report it instead of fixing package code.
- No test file in this tree may pull in torch; expectations are plain numpy.
