# HPO examples — runnable hyperparameter-optimization demos

## Intent
Runnable, CPU-only demo scripts for EvoX's hyperparameter optimization (HPO) framework, which turns an
inner `evox.core.Workflow` into an `evox.core.Problem` whose search space is the inner workflow's
hyperparameters. Each script is standalone (bootstraps `sys.path` to the repo root), runs in a few
seconds, prints human-readable progress, and doubles as a smoke test for the HPO path.

All scripts use the torch reference API (`evox.*`) because **HPO exists only in
`src/evox/problems/hpo_wrapper.py`; it was never ported to `src/evox_etl/`**.

## API Surface
| Script | Demonstrates | Inner workflow | Outer optimizer |
|---|---|---|---|
| `single_objective_hpo.py` | Single-objective HPO end-to-end, vs. untuned defaults | `PSO` on 5-D `Rastrigin`, `HPOFitnessMonitor()` | `PSO` over `(w, phi_p, phi_g)`, 15 generations |
| `multi_objective_hpo.py` | Multi-objective HPO scored by IGD to the DTLZ1 front | custom `(mu, lambda)` MO-EA on `DTLZ1(d=2, m=2)`, `HPOFitnessMonitor(multi_obj_metric=lambda f: igd(f, pf))` | `PSO` over `(sigma, p_m)`, 9 generations |
| `repeated_hpo.py` | `num_repeats > 1` repeated-evaluation / fitness-aggregation path | `PSO` on 5-D `Sphere`, `HPOFitnessMonitor(num_repeats=3)` | `PSO` over `(w, phi_p, phi_g)`, 5 generations |

Run command (from the repo root, CPU-only, no pytest/sklearn/optuna needed):
```
/home/bill/Source/evox/.venv/bin/python examples/hpo/<script>.py
```
Public framework surface used: `evox.problems.hpo_wrapper.{HPOProblemWrapper, HPOFitnessMonitor}`,
`evox.workflows.{StdWorkflow, EvalMonitor}`, `evox.metrics.igd`, `evox.problems.numerical.{Sphere, Rastrigin, DTLZ1}`,
`evox.algorithms.PSO`, `evox.core.{Algorithm, Mutable, Parameter, Problem}`, `evox.operators.selection`.

## Constraints
- **Public API only** — never import or call private/underscore framework members
  (`_hpo_evaluate_loop`, `_init_params`, `_workflow_init_step_`, `_stateful_tell_fitness`, `get_sub_state`, …)
  from an example. Mechanism explanations belong in comments/CONTEXT.md, not in internals re-implementations.
- Script conventions: module docstring with the exact run command, `sys.path` bootstrap locating the repo
  root via `src/evox/__init__.py`, `main()` under `if __name__ == "__main__":`,
  `torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")`.
- CPU-only, tiny populations/iteration counts (each script finishes in ≲1 s of real work), seeded where
  possible so the printed numbers are stable.
- No test framework: a clean exit code plus readable output is the verification. Never print hardcoded results.
- Do not modify `src/` or `unit_test/` for the sake of these examples.
- The `sys.path` bootstrap unavoidably trips ruff `E402`; this matches `examples/quickstart.py` (no `# noqa`).

## Notes for Agents
- The inner workflow's monitor **must** be an `HPOFitnessMonitor`, otherwise `HPOProblemWrapper` asserts.
- The outer algorithm's `pop_size` must equal `num_instances`, and the outer `StdWorkflow` must pass a
  `solution_transform` module returning the hyperparameter dict (e.g. `{"algorithm.w": x[:, 0], ...}`).
- Tunable keys are the inner workflow's `evox.core.Parameter` attributes, named by attribute path
  (`algorithm.w`, `algorithm.phi_g`, `algorithm.sigma`, …). Discover them at runtime with
  `hpo_prob.get_params_keys()` / `get_init_params()`; wrap replacements in `torch.nn.Parameter(..., requires_grad=False)`
  with `num_instances` rows.
- `HPOProblemWrapper.evaluate(params)` returns **one scalar per hyperparameter row** — shape `(num_instances,)`
  — for both single- and multi-objective inner workflows.
- A multi-objective inner workflow is collapsed to a single scalar by `multi_obj_metric`, so the OUTER problem is
  single-objective: use a single-objective outer algorithm (`PSO`/`DE`), not `NSGA2`. `PSO` cannot be the inner
  algorithm for a multi-objective problem (its `step` assumes a 1-D fitness) — hence the small custom MO-EA in
  `multi_objective_hpo.py`.
- With `num_repeats > 1` the monitor's `fit_aggregation` (default `_mean_fit_aggregation`) is applied to each
  inner step's fitness tensor *before* the running minimum is taken, so the aggregated score is on a different
  scale than the `num_repeats=1` score — compare the two only qualitatively (see the honest measurement in
  `repeated_hpo.py`), never as "the same quantity".

## Routing Table
| Area | Path | Description |
|---|---|---|
| Single-objective HPO | `./examples/hpo/single_objective_hpo.py` | Outer PSO tunes inner PSO hyperparameters on Rastrigin; prints tuned vs. default fitness |
| Multi-objective HPO | `./examples/hpo/multi_objective_hpo.py` | IGD-scored DTLZ1 inner run; outer PSO tunes mutation parameters |
| Repeated evaluation | `./examples/hpo/repeated_hpo.py` | `num_repeats=3` aggregation path: per-row scores, spread across seeds, short outer loop |
