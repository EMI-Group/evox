# Examples — runnable EvoX (evox_etl) demo scripts

## Intent
Standalone, runnable teaching scripts that demonstrate EvoX end-to-end on the `evoxc` compiler
(the functional `evox_etl` API: frozen config dataclasses + plain `init`/`step` functions).
Each script is self-contained: it bootstraps `sys.path` to the repo root, so it runs from a
checkout without installing anything (an editable install also works but is not required).
Scripts double as smoke tests: if `examples/` runs cleanly, the framework works.
Dependencies: `numpy` + `evoxc` only; Plotly is used through `EvalMonitor.plot`, which is
optional and degrades to a warning + `None` when Plotly is absent.

## API Surface
| Script | Demonstrates | Runtime (numpy backend) |
|---|---|---|
| `quickstart.py` | `evox_etl` PSO on 10-D `Sphere` via `StdWorkflow.run`; NSGA2 on 2-objective `DTLZ2` via `init`/`init_step`/`step` + `flush`, Pareto-front report and optional `EvalMonitor.plot` | ~1.5 s |
| `hpo/` | Host-side hyperparameter optimization with `evox_etl.problems.hpo_wrapper` (`HPOProblemWrapper`, `HPSlot`, `HPOProblemConfig`, `HPOFitnessMonitorConfig`, `random_search`): single-objective HPO, multi-objective HPO scored by IGD, repeated evaluation / `fit_aggregation` | 2–4 s per script |

All scripts run on `backend="numpy"`. Every `StdWorkflow`/`HPOProblemConfig` also takes
`backend="iree"` (or `"xla"`) to compile the same graphs on a compiled evoxc backend — a
one-word change, no API difference, but those backends need their evoxc extra
(`pip install "evoxc[iree]"`); without it venv `evoxc` raises a `BackendError`.

## Constraints
- One script per topic, small and focused (well under ~1000 lines), heavily commented for learners.
- Each script: short module docstring with the exact run command (`python examples/<path>.py`),
  a `sys.path` bootstrap that walks up `pathlib.Path(__file__).resolve().parents` until it finds
  the directory containing `src/evox_etl/__init__.py`, then a `main()` under
  `if __name__ == "__main__":`.
- Scripts must run on **CPU only** and finish in a few seconds (tiny populations/iteration counts);
  the whole `examples/` tree runs in under ~12 s.
- No torch, no legacy `evox.*` API: scripts import only `numpy`, `evoxc` and `evox_etl`.
- The bootstrap unavoidably triggers ruff `E402` (imports after code). This matches the existing
  repo convention — the `examples/` tree carries no `# noqa` markers and the repo-wide ruff CI is
  expected to be run with `examples/` out of scope.
- No test framework here: a script's successful exit code and readable printed output are its
  verification. Printed numbers are measured, never hardcoded; the scripts also `assert` that
  re-evaluating a reported result reproduces it.
- Examples must never import private/underscore framework internals — public API only.

## Routing Table
| Area | Path | Description |
|---|---|---|
| Quickstart (functional ETL) | `./examples/quickstart.py` | PSO-on-Sphere (`StdWorkflow.run`) + NSGA2-on-DTLZ2 (manual `init`/`step`, Pareto front, optional Plotly figure) |
| Hyperparameter optimization | `./examples/hpo/` | Runnable HPO examples (host-side `HPOProblemWrapper` + `random_search`, numpy backend): SO HPO, MO HPO with IGD, `num_repeats > 1` aggregation |
