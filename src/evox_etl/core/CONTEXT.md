# evox_etl/core — protocols, state helpers, workflow

## Intent
The functional foundation of evox_etl: duck-typed protocol documentation (Algorithm:
`init/ask/tell`; Problem: `evaluate`; Monitor: `monitor_update`), `WorkflowState`
dataclass, state helpers, and `StdWorkflow` (compile-once step loop). See
`../DESIGN.md` §4 — the binding spec.

## Routing Table
| Area | Path | Notes |
|---|---|---|
| Algorithm/problem/monitor protocols | `algorithm.py`, `problem.py`, `monitor.py` | docstrings + typing only |
| State helpers | `state.py` | pytree/replace helpers |
| StdWorkflow | `workflow.py` | compose+compile+run loop |

## Notes for Agents
- Reference for workflow behavior: `../../evox/workflows/std_workflow.py` (torch,
  read-only sibling) and pre-1.0 JAX version via `git show v0.9.0:src/evox/workflows/std_workflow.py`.
- ETL same-device loop pattern is validated in etl tests
  (`tests/backends/test_iree_same_device_loop.py` in the foreign repo).
