# Examples — runnable EvoX demo scripts

## Intent
Standalone, runnable teaching scripts that demonstrate EvoX end-to-end.
Each script is self-contained: it bootstraps `sys.path` to the repo root, so it runs from a
checkout without installing anything (the editable install of `evox` also works but is not required).
Scripts double as smoke tests: if `examples/` runs cleanly, the framework works.

## API Surface
| Script | Demonstrates |
|---|---|
| `quickstart.py` | `evox_etl` (numpy backend): PSO on Sphere, `EvalMonitorConfig`, `StdWorkflow` |
| `hpo/` | Hyperparameter optimization with the torch reference API (`evox.*`): single-objective HPO, multi-objective HPO (IGD metric), repeated evaluation (`num_repeats > 1`) |

## Constraints
- One script per topic, small and focused (well under ~1000 lines), heavily commented for learners.
- Each script: short module docstring with the exact run command, a `sys.path` bootstrap that walks up
  `pathlib.Path(__file__).resolve().parents` until it finds the directory containing
  `src/<package>/__init__.py`, then a `main()` under `if __name__ == "__main__":`.
- Scripts must run on **CPU only** and finish in a few seconds (tiny populations/iteration counts).
- The bootstrap unavoidably triggers ruff `E402` (imports after code). This matches the existing
  repo convention in `examples/quickstart.py` — the `examples/` tree carries no `# noqa` markers and
  the repo-wide ruff CI is expected to be run with `examples/` out of scope.
- No test framework here: a script's successful exit code and readable printed output are its verification.
- Examples must never import private/underscore framework internals — public API only.

## Routing Table
| Area | Path | Description |
|---|---|---|
| Quickstart (functional ETL, numpy) | `./examples/quickstart.py` | Minimal PSO-on-Sphere end-to-end demo of `evox_etl` |
| Hyperparameter optimization | `./examples/hpo/` | Runnable HPO examples (torch reference API): SO HPO, MO HPO with IGD, `num_repeats > 1` aggregation |
