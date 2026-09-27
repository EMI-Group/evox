# unit_test/etl/workflows — workflows suite
## Intent
pytest suite for `evox_etl.workflows` (EvalMonitor semantics + StdWorkflow
monitor/history behavior). Numpy backend only — no torch imports. Runnable
without installing `evox_etl` (it is a PEP 420 namespace package under `src/`).
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
- `test_eval_monitor_plot.py` — end-to-end tests for the implemented `EvalMonitor.plot`
  (Plotly figures): SO 1-D (PSO+Sphere, `animation` kwarg forwarding, "eval"
  source un-negating `opt_direction="max"`), MO 2-D/3-D (NSGA2+DTLZ2 with and
  without `problem_pf`), the POP source via the toy aux `"fit"` channel (raw,
  NOT un-negated), the warn+None paths (no history, Plotly absent by
  monkeypatching `evox_etl.vis_tools.plot.go = None`, ≥4 objectives) and the
  invalid-`source` ValueError. Figure assertions are `skipif`-guarded on plotly.
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
- Run: `PYTHONPATH=src:<site-packages> <venv>/bin/python -m pytest
  unit_test/etl/workflows -q` (38 tests; ~14 s with plotly available).
