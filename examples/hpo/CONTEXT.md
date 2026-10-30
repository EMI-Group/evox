# HPO examples — runnable hyperparameter-optimization demos (evox_etl)

## Intent
Runnable, CPU-only demo scripts for EvoX's hyperparameter optimization (HPO) on the `evoxc`
compiler. They drive the **host-side** HPO redesign of `src/evox_etl/problems/hpo_wrapper.py`:
`HPOProblemWrapper` holds the *inner* algorithm/problem configs plus an `HPSlot` spec of which
inner-config fields are tuned, and `evaluate(candidate)` is plain host Python that substitutes the
values (`dataclasses.replace`), builds a fresh inner `StdWorkflow`, runs it (`wf.run(gens, seed=)`)
and reads the inner monitor's best fitness. `random_search` is the minimal outer search driver.
The wrapper is deliberately NOT an `evox_etl` `Problem` — a nested compiled workflow can never run
inside an outer trace (`evoxc` has no eager mode) — so the outer loop is host-side Python.

Each script is standalone (bootstraps `sys.path` to the repo root), runs in a few seconds, prints
human-readable progress, and doubles as a smoke test for the HPO path. Dependencies: `numpy` +
`evoxc` only, numpy backend.

## API Surface
| Script | Demonstrates | Inner run | Search | Runtime |
|---|---|---|---|---|
| `single_objective_hpo.py` | Single-objective HPO vs. untuned defaults, then building a `StdWorkflow` from the tuned config | PSO on 10-D `Rastrigin`, 30 generations, `HPOFitnessMonitorConfig(inner_monitor=EvalMonitorConfig())` | `random_search` over `(w, phi_p, phi_g)`, 18 trials | ~2.5 s |
| `multi_objective_hpo.py` | Multi-objective HPO scored by IGD to the true front (`multi_obj_metric`), `opt_direction=["min", "min"]`, `full_fit_history=True` | RVEA on 2-objective `DTLZ2` (d=5), 30 generations | `random_search` over `(alpha, fr)`, 8 trials | ~4 s |
| `repeated_hpo.py` | Seed spread, `num_repeats > 1` and every `fit_aggregation` (`"mean"`/`"min"`/`"max"`), seed-averaged search | PSO on 10-D `Rastrigin`, 25 generations, `num_repeats=3` | `random_search` over `(w, phi_p, phi_g)`, 12 trials | ~3 s |

Run command (from the repo root, CPU-only, no pytest/sklearn/optuna needed):
```
python examples/hpo/<script>.py
```
Public framework surface used: `evox_etl.problems.hpo_wrapper.{HPOFitnessMonitorConfig,
HPOProblemConfig, HPOProblemWrapper, HPSlot, random_search}`, `evox_etl.{StdWorkflow,
EvalMonitorConfig}`, `evox_etl.algorithms.{PSO, RVEA, NSGA2}`, `evox_etl.metrics.igd`,
`evox_etl.problems.numerical.{Rastrigin, DTLZ2}` (+ `DTLZ`.pf for the reference front),
`evoxc.build`/`evoxc.core.TensorSpec` for the host-side metric wrapper.

## Constraints
- **Public API only** — never import or call private/underscore framework members
  (`_build_workflow`, `_resolve_values`, `_fresh_inner_monitor`, `_defaults`, …) from an example.
  Mechanism explanations belong in comments/CONTEXT.md, not in internals re-implementations.
- Script conventions: module docstring with the exact run command, `sys.path` bootstrap locating
  the repo root via `src/evox_etl/__init__.py`, `main()` under `if __name__ == "__main__":`,
  `backend="numpy"` (documented as switchable to `"iree"`/`"xla"`, which need the evoxc
  backend extra, e.g. `pip install "evoxc[iree]"`).
- CPU-only, tiny populations/generation counts (each script finishes in a few seconds), seeds
  fixed so the printed numbers are stable and the searches are deterministic.
- No test framework: a clean exit code plus readable output is the verification. Never print
  hardcoded results; the scripts `assert` that re-evaluating a reported candidate reproduces it.
- Do not modify `src/` or `unit_test/` for the sake of these examples.
- The `sys.path` bootstrap unavoidably trips ruff `E402`; this matches `examples/quickstart.py`
  (no `# noqa`).

## Notes for Agents
- Tunable fields are addressed by `HPSlot(config_kind, field)` with `config_kind` in
  `"algorithm"` / `"problem"` / `"monitor"`, and by the dotted key `<kind>.<field>` afterwards.
  Discover them at runtime with `wrapper.get_params_keys()` / `wrapper.get_init_params()`; an
  invalid field is rejected at construction time with the list of available fields.
- Hyperparameters must be plain STATIC config values (Python scalars, tuples, strings) —
  `dataclasses.replace` substitutes them; tensor-valued fields are unsupported. Integer fields
  need an explicit `transform` (e.g. `HPSlot("algorithm", "pop_size", transform=lambda v: int(round(v)))`).
- `evaluate(None)` / `evaluate({})` evaluates the inner configs unchanged — that is the
  "defaults" baseline. Missing mapping keys fall back to the inner config's value, and a raw
  vector is read in `HPSlot` order; `evaluate_many` loops over rows.
- The inner monitor must be an `EvalMonitorConfig` (read through `HPOFitnessMonitorConfig`), and a
  fresh copy with empty history containers is built per candidate.
- Multi-objective inner runs need `multi_obj_metric` (the single-objective `get_best_fitness`
  accessor is SO-only) plus `full_fit_history=True` and an `opt_direction` LIST. `multi_obj_metric`
  is a HOST-side callable over an `(n_pareto, n_objs)` numpy array; an evoxc metric such as
  `evox_etl.metrics.igd` must first be wrapped host-side with `evoxc.build` + `evoxc.run`
  (see `HostIGD` in `multi_objective_hpo.py`), cached per input shape because the front size varies.
- `num_repeats > 1`: repeat `r` runs with seed `config.seed + r`; the per-repeat best fitness values
  are combined by `fit_aggregation`. `"mean"` is the seed-robust score to optimize, `"min"`/`"max"`
  are the optimistic/pessimistic bounds of the spread. Reusing a workflow across repeats is safe
  because the monitor's elite is re-initialized by every `run` (only `get_pf_fitness`'s host-side
  history accumulates, so MO examples use `num_repeats=1`).
- Every distinct candidate re-traces and re-compiles the inner step graph (config values are static
  trace leaves), so per-candidate cost is dominated by compile time — keep populations and
  generation counts small in examples.
- Choose inner problems where the defaults are genuinely suboptimal (10-D Rastrigin rather than
  Sphere; DTLZ2 rather than the much harder DTLZ1), otherwise a small random search finds nothing
  better than the defaults and the demo prints a discouraging result.

## Routing Table
| Area | Path | Description |
|---|---|---|
| Single-objective HPO | `./examples/hpo/single_objective_hpo.py` | `random_search` tunes inner PSO weights on 10-D Rastrigin; prints tuned vs. default fitness and rebuilds the tuned `StdWorkflow` |
| Multi-objective HPO | `./examples/hpo/multi_objective_hpo.py` | IGD-scored RVEA-on-DTLZ2 inner run (host-side `evoxc.build` metric wrapper); tunes `(alpha, fr)`, prints the IGD reduction |
| Repeated evaluation | `./examples/hpo/repeated_hpo.py` | `num_repeats=3` seed spread + `fit_aggregation` policies, then a seed-averaged search |
