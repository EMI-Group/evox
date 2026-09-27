# src/evox_etl/workflows/tests — in-node pytest suite

## Intent
pytest suite for `evox_etl.workflows` (EvalMonitor semantics + StdWorkflow
monitor/history behavior), runnable without installing `evox_etl` (it is a PEP
420 namespace package under `src/`). Numpy backend only — no torch imports.

NOTE (relocation): these files live here while the migration wave settles; each
test file is written to be moved to `unit_test/etl/workflows/` verbatim.

## Files
- `conftest.py` — sys.path shim (repo root + `src/`), idempotent, locations
  computed from `__file__`. Shared bootstrap; do NOT add a second conftest.
- `test_eval_monitor.py` — module-level `init`/`monitor_update` shape policy
  (SO wide/narrow batches, MO, wrapper accessors) driven directly through
  `etl.build`/`etl.run`, independent of `StdWorkflow`.
- `test_workflow_monitor_shapes.py` — end-to-end `StdWorkflow` tests for
  monitor-batch shape drift using real algorithms (CoDE `3n`, CSO `n/2`,
  PSO `n`).
- `test_aux_history.py` — end-to-end tests for the optional auxiliary-history
  channel: `algorithm record_step` -> `monitor record_auxiliary` ->
  `EvalMonitorConfig.aux_history` (per-generation numpy entries + shapes),
  `EvalMonitor.pop_history` preferring `aux_history["pop"]`, the
  `full_pop_history` gate, and the no-hook legacy back-compat path. Also
  unit-tests the module-level `record_auxiliary` gate/accumulation.
- Toy components for `test_aux_history.py`, one config per module (the workflow
  resolves each component's plain functions via `type(config).__module__`):
  `aux_toy_algorithm.py` (WITH `record_step`), `aux_toy_algorithm_no_hook.py`
  (WITHOUT `record_step`), `aux_toy_problem.py` (stateless sphere).

## Constraints
- NO `__init__.py` here: pytest's prepend import mode imports the test modules
  AND the `aux_toy_*` siblings as TOP-LEVEL modules, which is exactly what makes
  `type(config).__module__` resolvable via `importlib`. New helper/toy modules
  need distinctive basenames to avoid collisions with other test directories.
- Two configs used together by one workflow MUST live in distinct modules.
- The directory is gitignored (`.gitignore:168 "tests"`); new files must be
  `git add -f`'d to be tracked, like the pre-existing ones.
- Run: `PYTHONPATH=src:<site-packages> <venv>/bin/python -m pytest
  src/evox_etl/workflows/tests -q` (24 tests: 9 aux-history + 15 pre-existing).
